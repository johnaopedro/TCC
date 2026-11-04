import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ALGORITHMS = ("threshold", "equalization", "sobel-prewitt-roberts")
DIMENSIONS = {"base": (500, 500), "media": (1024, 1024), "alta": (2048, 2048),
              "uhd4k": (3840, 2160), "uhd8k": (7680, 4320)}
IMAGES = {"lena": ROOT / "tcc-images/lena.jpg", "sudoku": ROOT / "tcc-images/sudoku.png",
          "building": ROOT / "tcc-images/building.jpg"}
THREADS = (1, 2, 4, 6, 8)
REPETITIONS = 3
SEED = 20261008
FLAGS = ["-O3", "-std=c++17", "-ffp-contract=off", "-Wno-unknown-pragmas"]
TIME = re.compile(r"^.* time: ([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?) seconds$", re.M)
RSS = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")


def now():
    return datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()


def digest(path):
    with path.open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def write_json(path, data):
    # Only newly created run artifacts are written; algorithms are never edited.
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command, env=None):
    return subprocess.run([str(x) for x in command], env=env, cwd=ROOT, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout


def read_optional(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def topology():
    groups = defaultdict(list)
    details = []
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        socket = int((directory / "physical_package_id").read_text())
        core = int((directory / "core_id").read_text())
        groups[(socket, core)].append(cpu)
        details.append(dict(cpu=cpu, socket=socket, core=core,
                            siblings=(directory / "thread_siblings_list").read_text().strip()))
    representatives = [min(cpus) for _, cpus in sorted(groups.items())]
    remaining = [cpu for _, members in sorted(groups.items()) for cpu in sorted(members)
                 if cpu not in representatives]
    return dict(allowed=details, physical_cores_available=len(groups),
                representatives=representatives,
                logical_cpus_available=len(details), logical_cpu_order=representatives + remaining,
                smt_active=read_optional("/sys/devices/system/cpu/smt/active"),
                smt_control=read_optional("/sys/devices/system/cpu/smt/control"))


def select_threads(mode, requested, hardware):
    if not requested or len(set(requested)) != len(requested) or any(p not in THREADS for p in requested):
        raise ValueError("threads devem ser valores distintos de 1, 2, 4, 6 e 8")
    if mode == "official":
        if tuple(requested) != THREADS:
            raise ValueError("a coleta oficial exige exatamente 1, 2, 4, 6 e 8 threads")
        if hardware["physical_cores_available"] < 8:
            raise ValueError("coleta oficial exige pelo menos 8 núcleos físicos disponíveis; "
                             f"detectados {hardware['physical_cores_available']}. Não usar SMT nem sobreassinatura")
        # Kernel reports 0 for disabled or unsupported SMT. Unknown is not proof.
        if hardware["smt_active"] != "0":
            raise ValueError("coleta oficial exige HT/SMT inativo (smt/active=0). "
                             "Não desativamos configurações do sistema automaticamente")
        return list(requested), []
    if mode == "verify":
        return [], []
    if mode == "local-ht":
        if max(requested) > hardware["logical_cpus_available"]:
            raise ValueError("faltam processadores lógicos disponíveis; não usar sobreassinatura "
                             "além das threads de hardware")
        return list(requested), []
    selected = [p for p in requested if p <= hardware["physical_cores_available"]]
    skipped = [p for p in requested if p not in selected]
    if not selected:
        raise ValueError("nenhuma configuração cabe nos núcleos físicos disponíveis")
    return selected, skipped


def measurement(stdout, stderr, count):
    times = [float(x) for x in TIME.findall(stdout)]
    memory = [int(x) for x in RSS.findall(stderr)]
    if len(times) != count or len(memory) != count:
        raise RuntimeError(f"esperadas {count} medições de núcleo e GNU time; "
                           f"obtidas {len(times)} e {len(memory)}")
    if any(not math.isfinite(x) or x <= 0 for x in times):
        raise RuntimeError("tempo ausente, não positivo ou não finito")
    return times, memory


def thread_environment(environment, cpus, p):
    if len(cpus) < p or len(set(cpus)) != len(cpus):
        raise ValueError("afinidade exige um processador lógico distinto por thread")
    return dict(environment, OMP_NUM_THREADS=str(p),
                OMP_PLACES=",".join("{" + str(cpu) + "}" for cpu in cpus[:p]))


def configurations(timing_threads):
    return [(a, i, r, v, p) for a in ALGORITHMS for i in IMAGES for r in DIMENSIONS
            for v, p in [("sequencial", 1)] + [("paralelizada", p) for p in timing_threads]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official", "local", "local-ht", "verify"), default="local-ht",
                        help="padrão local-ht permite SMT; official exige HT inativo e 8 núcleos; "
                             "local mede apenas p <= núcleos; verify não mede")
    parser.add_argument("--threads", nargs="+", type=int, default=list(THREADS))
    args = parser.parse_args()
    hardware = topology()
    try:
        timing_threads, skipped = select_threads(args.mode, args.threads, hardware)
    except ValueError as error:
        print("ERRO: " + str(error), file=sys.stderr)
        return 2
    required = ["g++", "pkg-config", "taskset"] + (["perf"] if args.mode == "official" else [])
    for dependency in required:
        if shutil.which(dependency) is None:
            raise RuntimeError("dependência ausente: " + dependency)
    if not Path("/usr/bin/time").is_file():
        raise RuntimeError("GNU time ausente em /usr/bin/time")
    if args.mode == "official":
        # Read-only capability check; never change kernel permissions.
        probe = subprocess.run(["perf", "stat", "-d", "--", "true"], text=True, capture_output=True)
        if probe.returncode:
            raise RuntimeError("perf indisponível; solicitar ao responsável pela máquina: " + probe.stderr)
    (ROOT / "resultados").mkdir(exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=now()[:10] + "-", dir=ROOT / "resultados"))
    print("EXECUCAO=" + str(output), flush=True)
    if skipped:
        print(f"AVISO: tempos p={skipped} omitidos; apenas {hardware['physical_cores_available']} "
              "núcleos físicos. Corretude será testada com 1,2,4,6,8 independentemente disso", flush=True)
    cpu_order = (hardware["logical_cpu_order"] if args.mode == "local-ht" else hardware["representatives"])
    cpus = cpu_order[:max(timing_threads or [1])]
    env = os.environ.copy()
    # Singleton places document exactly which logical CPUs are used, including SMT when requested.
    env.update(OMP_DYNAMIC="false", OMP_PLACES=",".join("{" + str(c) + "}" for c in cpus),
               OMP_PROC_BIND="close", OMP_WAIT_POLICY="PASSIVE", LC_ALL="C")
    metadata = dict(status="INCOMPLETE", start=now(), mode=args.mode, topology=hardware, timing_threads=timing_threads,
                    requested_timing_threads=args.threads, skipped_timing_threads=skipped,
                    placement_policy="physical cores first, then SMT siblings" if args.mode == "local-ht" else
                                     "one logical CPU per distinct physical core",
                    correctness_threads=list(THREADS), warmup=1, repetitions=REPETITIONS, random_seed=SEED,
                    dimensions=DIMENSIONS, flags=FLAGS, affinity_cpus=cpus,
                    omp={key: env[key] for key in env if key.startswith("OMP_")},
                    primary_comparison="sequencial adaptado versus versão paralelizada; ganho total",
                    compiler=run(["g++", "--version"]), cpu=run(["lscpu"]),
                    opencv=run(["pkg-config", "--modversion", "opencv4"]).strip(),
                    kernel=run(["uname", "-a"]).strip(),
                    load_start=read_optional("/proc/loadavg"), memory_start=read_optional("/proc/meminfo"),
                    perf_event_paranoid=read_optional("/proc/sys/kernel/perf_event_paranoid"),
                    sources={}, inputs={}, binaries={})
    workspace = tempfile.TemporaryDirectory(prefix="tcc-benchmark-")
    work_dir = Path(workspace.name)
    bin_dir, logs = work_dir / "bin", work_dir / "logs"
    bin_dir.mkdir()
    logs.mkdir()
    cv = shlex.split(run(["pkg-config", "--cflags", "--libs", "opencv4"]).strip())
    metadata["opencv_build_flags"] = cv
    write_json(output / "environment.json", metadata)
    try:
        sources = [ROOT / "paralelizados" / (a + ".cpp") for a in ALGORITHMS]
        sources += [ROOT / "originais" / (a + ".cpp") for a in ALGORITHMS]
        sources += [ROOT / "paralelizados/common/bench.hpp", ROOT / "originais/common/bench.hpp",
                    ROOT / "scripts/benchmark.sh", HERE / "prepare_images.cpp", ROOT / "testes/verificar_entradas.cpp",
                    HERE / "afinidade.cpp", Path(__file__).resolve(), ROOT / "testes/verify.cpp",
                    ROOT / "testes/reference.hpp",
                    ROOT / "testes/referencias.inc"]
        for source in sources:
            relative = source.relative_to(ROOT)
            metadata["sources"][str(relative)] = digest(source)
        binaries = {}
        for algorithm in ALGORITHMS:
            for variant in ("sequencial", "paralelizada"):
                source = (ROOT / "originais" / (algorithm + ".cpp")
                          if variant == "sequencial" else ROOT / "paralelizados" / (algorithm + ".cpp"))
                executable = bin_dir / (variant + "__" + algorithm)
                print(f"Compilando {algorithm}: {variant}", flush=True)
                run(["g++", *FLAGS, *(["-fopenmp"] if variant == "paralelizada" else []),
                     source, *cv, "-o", executable])
                binaries[(variant, algorithm)] = executable
                metadata["binaries"][executable.name] = digest(executable)
        helpers = {}
        for name in ("preparar", "afinidade"):
            helpers[name] = bin_dir / name
            run(["g++", *FLAGS, *(["-fopenmp"] if name == "afinidade" else []),
                 HERE / ("prepare_images.cpp" if name == "preparar" else "afinidade.cpp"), *cv, "-o", helpers[name]])
        metadata["affinity_checks"] = []
        core_map = {entry["cpu"]: (entry["socket"], entry["core"]) for entry in hardware["allowed"]}
        for p in timing_threads:
            pinned = cpus[:p]
            check = json.loads(run(["taskset", "-c", ",".join(map(str, pinned)), helpers["afinidade"], p],
                                   env=thread_environment(env, cpus, p)))
            if check["actual"] != p or len(set(check["cpus"])) != p or any(c not in pinned for c in check["cpus"]):
                raise RuntimeError("afinidade não corresponde aos processadores lógicos selecionados")
            cores_used = {core_map[cpu] for cpu in check["cpus"]}
            if len(cores_used) != min(p, hardware["physical_cores_available"]):
                raise RuntimeError("afinidade não distribuiu primeiro pelos núcleos físicos")
            check["physical_cores_used"] = len(cores_used)
            check["uses_smt_siblings"] = len(cores_used) < p
            metadata["affinity_checks"].append(check)
        print("Afinidade confirmada; CPUs lógicas e núcleos físicos registrados por configuração", flush=True)
        inputs = {}
        for image, source in IMAGES.items():
            directory = work_dir / "inputs" / image
            run([helpers["preparar"], source, directory])
            proof = run([helpers["preparar"], "--verify-source", source, directory])
            (logs / ("origem-" + image + ".log")).write_text(proof)
            metadata["inputs"][str(source.relative_to(ROOT))] = digest(source)
            for resolution in DIMENSIONS:
                path = directory / (resolution + ".png")
                inputs[(image, resolution)] = path
                metadata["inputs"][f"inputs/{image}/{resolution}.png"] = digest(path)
        print("15 entradas preparadas e verificadas, incluindo três imagens 8K", flush=True)
        # Broad regression suite includes float intermediates. SMT does not compromise
        # correctness comparisons. Timed use of SMT is explicit in mode local-ht.
        verification_env = dict(env, OMP_NUM_THREADS="1", OMP_PLACES="threads")
        executable = bin_dir / "casos_par"
        run(["g++", *FLAGS, "-DEXP_PAR=1", "-fopenmp", ROOT / "testes/verify.cpp",
             *cv, "-o", executable])
        log = run([executable], env=verification_env)
        (logs / "casos_par.log").write_text(log)
        if "10560 matrizes identicas" not in log:
            raise RuntimeError("regressão de 44 entradas não cobriu a matriz esperada")
        print(log.splitlines()[-1], flush=True)
        comparisons = 0
        executable = bin_dir / "entradas_par"
        run(["g++", *FLAGS, "-DEXP_PAR=0", "-DBASE_PAR_TEST=1", "-fopenmp",
             ROOT / "testes/verificar_entradas.cpp", *cv, "-o", executable])
        for p in THREADS:
            print(f"Validando 15 entradas, versão paralelizada, p={p}", flush=True)
            log = run([executable, p, *inputs.values()], env=dict(verification_env, OMP_NUM_THREADS=str(p)))
            (logs / f"entradas_par-p{p}.log").write_text(log)
            if "15 entradas, 225 saidas referencia/paralelizada identicas byte a byte." not in log:
                raise RuntimeError("validação das entradas preparada incompleta")
            comparisons += 225
            print(log.splitlines()[-1], flush=True)
        metadata["prepared_output_comparisons"] = comparisons
        # Real CLI integration smoke: saves only outside timed experiments.
        smoke = work_dir / "validacao-cli"
        for variant in ("sequencial", "paralelizada"):
            directory = smoke / variant
            directory.mkdir(parents=True)
            for algorithm in ALGORITHMS:
                run([binaries[(variant, algorithm)], inputs[("lena", "uhd8k")], directory], env=env)
        pngs = sorted((smoke / "sequencial").glob("*.png"))
        if len(pngs) != 15 or any(digest(p) != digest(smoke / "paralelizada" / p.name) for p in pngs):
            raise RuntimeError("CLI real produziu arquivos PNG divergentes")
        metadata["cli_8k_png_comparisons"] = 15
        print("15 PNGs dos executáveis reais em 8K idênticos (p=1)", flush=True)
        write_json(output / "environment.json", metadata)
        if args.mode == "verify":
            for source, before in metadata["sources"].items():
                if digest(ROOT / source) != before:
                    raise RuntimeError("fonte mudou durante a validação: " + source)
            metadata.update(finish=now(), status="VERIFIED_NO_TIMING")
            write_json(output / "environment.json", metadata)
            print("CORRETUDE CONCLUÍDA; nenhum tempo de benchmark coletado", flush=True)
            return 0

        configs = configurations(timing_threads)
        rng = random.Random(SEED)
        warmups = list(configs)
        rng.shuffle(warmups)
        samples = []
        with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
            def execute(config, phase, repetition, position, count=1):
                a, i, r, variant, p = config
                pinned = cpus[:p] if variant == "paralelizada" else cpus[:1]
                command = ["taskset", "-c", ",".join(map(str, pinned)), "/usr/bin/time", "-v",
                           str(binaries[(variant, a)]), str(inputs[(i, r)])]
                process_env = thread_environment(env, cpus, p)
                name = f"{phase}-{position:04d}-{a}-{i}-{r}-{variant}-p{p}"
                if count == 3:
                    command = ["perf", "stat", "-d", "-r", "3", "-o", str(logs / (name + "-perf.txt")),
                               "--", *command]
                process = subprocess.run(command, env=process_env, text=True, capture_output=True, check=True)
                times, memory = measurement(process.stdout, process.stderr, count)
                for index, (seconds, rss) in enumerate(zip(times, memory)):
                    row = dict(algorithm=a, image=i, resolution=r, variant=variant, threads=p,
                               phase=phase, position=position, repetition=repetition + index,
                               seconds=seconds, max_rss_kib=rss, command=command)
                    if phase == "measured" and index == 0:
                        row["process_reports"] = {"gnu_time": process.stderr}
                        if count == 3:
                            row["process_reports"]["perf_stat"] = (logs / (name + "-perf.txt")).read_text()
                            row["process_reports"]["perf_scope"] = "agregado das tres repeticoes do processo completo"
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    if phase == "measured":
                        samples.append(row)
            print(f"Aquecimento: {len(warmups)} configurações", flush=True)
            for position, config in enumerate(warmups, 1):
                execute(config, "warmup", 0, position)
                if position % 25 == 0 or position == len(warmups):
                    print(f"warmup {position}/{len(warmups)}", flush=True)
            jobs = ([(c, 1) for c in configs] if args.mode == "official" else
                    [(c, rep) for c in configs for rep in (1, 2, 3)])
            rng.shuffle(jobs)
            for position, (config, repetition) in enumerate(jobs, 1):
                execute(config, "measured", repetition, position, 3 if args.mode == "official" else 1)
                if position % 25 == 0 or position == len(jobs):
                    print(f"measured {position}/{len(jobs)}", flush=True)
        high_cv_count = 0
        for a, i, r, variant, p in configs:
            values = [row["seconds"] for row in samples if
                      (row["algorithm"], row["image"], row["resolution"], row["variant"], row["threads"]) ==
                      (a, i, r, variant, p)]
            if len(values) != REPETITIONS:
                raise RuntimeError("amostras incompletas")
            if 100 * statistics.stdev(values) / statistics.mean(values) > 5:
                high_cv_count += 1
        for source, before in metadata["sources"].items():
            if digest(ROOT / source) != before:
                raise RuntimeError("fonte mudou durante a execução: " + source)
        metadata.update(finish=now(), load_end=read_optional("/proc/loadavg"), memory_end=read_optional("/proc/meminfo"),
                        configurations=len(configs), valid_samples=len(samples),
                        high_cv_count=high_cv_count)
        status = "COMPLETE_OFFICIAL" if args.mode == "official" else (
            "COMPLETE_LOCAL_HT" if args.mode == "local-ht" else "COMPLETE_EXPLORATORY")
        metadata["status"] = status
        write_json(output / "environment.json", metadata)
        print(f"CONCLUÍDO: {len(samples)} amostras, {len(configs)} configurações; "
              f"{metadata['high_cv_count']} com CV>5%", flush=True)
        print("DADOS=" + str(output / "results.jsonl"), flush=True)
        return 0
    except Exception as error:
        metadata.update(failure_time=now(), status="FAILED",
                        failure={"message": str(error), "stdout": getattr(error, "stdout", None),
                                 "stderr": getattr(error, "stderr", None)})
        write_json(output / "environment.json", metadata)
        raise
    finally:
        workspace.cleanup()


if __name__ == "__main__":
    sys.exit(main())
