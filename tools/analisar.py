import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = {
    "threshold": "Threshold",
    "equalization": "Equalização",
    "dilation-erosion": "Dilatação e erosão",
    "linear-interpolation": "Interpolação",
    "sobel-prewitt-roberts": "Bordas",
}
REFACTORINGS = {
    "threshold": "Cinco modalidades em uma varredura",
    "equalization": "LUT de 256 posições",
    "sobel-prewitt-roberts": "Redução conjunta dos extremos",
}
RESOLUTIONS = {"base": "Base", "media": "Média", "alta": "Alta",
               "uhd4k": "4K UHD", "uhd8k": "8K UHD"}
MEGAPIXELS = {"base": 0.25, "media": 1.05, "alta": 4.19, "uhd4k": 8.29, "uhd8k": 33.18}
IMAGES = {"lena": "Lena", "sudoku": "Sudoku", "building": "Building"}
THREADS = (1, 2, 4, 6, 8)
BUDGETS_MS = (33.3, 16.7)
CV_LIMIT = 5.0

GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
GROUP_MARKERS = ["o", "s", "^", "D", "v"]
RES_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
RES_MARKERS = ["o", "s", "^", "D", "v"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def br(value, digits=2):
    if value is None:
        return "--"
    return f"{value:.{digits}f}".replace(".", ",")


def esc(text):
    table = {"\\": "\\textbackslash{}", "_": "\\_", "%": "\\%", "&": "\\&", "#": "\\#",
             "$": "\\$", "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}", "^": "\\textasciicircum{}"}
    return "".join(table.get(c, c) for c in str(text))


def field(text, *names):
    for line in (text or "").splitlines():
        key, _, value = line.partition(":") if ":" in line else line.partition("=")
        if key.strip().lower() in [n.lower() for n in names]:
            return value.strip().strip('"')
    return None


def ordered(keys, order):
    known = [k for k in order if k in keys]
    return known + sorted(k for k in keys if k not in order)


def load(run):
    rows = []
    if not (run / "results.jsonl").exists():
        sys.exit(f"ERRO: {run}/results.jsonl nao existe (o modo verify nao coleta tempos)")
    with (run / "results.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("phase") == "measured" and row.get("variant") in ("sequencial", "paralelizada"):
                rows.append(row)
    if not rows:
        sys.exit("ERRO: nenhuma medicao (phase = measured) em results.jsonl")
    env_path = run / "environment.json"
    env = json.loads(env_path.read_text(encoding="utf-8")) if env_path.exists() else {}
    return rows, env


def aggregate(rows):
    samples = defaultdict(list)
    for r in rows:
        key = (r["algorithm"], r["image"], r["resolution"], r["variant"],
               int(r["threads"]), r.get("schedule", "static"))
        samples[key].append(1000.0 * float(r["seconds"]))
    stats = {}
    for key, values in samples.items():
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[key] = dict(n=len(values), mean=mean, sd=sd, cv=100.0 * sd / mean if mean else 0.0)
    return stats


class Data:
    def __init__(self, stats):
        self.stats = stats
        self.groups = ordered({k[0] for k in stats}, GROUPS)
        self.images = ordered({k[1] for k in stats}, IMAGES)
        self.resolutions = ordered({k[2] for k in stats}, RESOLUTIONS)

    def t(self, g, img, res, p=None, schedule="static"):
        key = (g, img, res, "sequencial", 1, "static") if p is None else \
              (g, img, res, "paralelizada", p, schedule)
        s = self.stats.get(key)
        return s["mean"] if s else None

    def per_image(self, g, res, func):
        values = [func(img) for img in self.images]
        values = [v for v in values if v is not None]
        return statistics.mean(values) if values else None

    def ts(self, g, res):
        return self.per_image(g, res, lambda img: self.t(g, img, res))

    def tp(self, g, res, p):
        return self.per_image(g, res, lambda img: self.t(g, img, res, p))

    def speedup(self, g, res, p):
        def one(img):
            ts, tp = self.t(g, img, res), self.t(g, img, res, p)
            return ts / tp if ts and tp else None
        return self.per_image(g, res, one)

    def internal(self, g, res, p):
        """T_1/T_p: escalabilidade da versão paralelizada em relação a ela mesma."""
        def one(img):
            t1, tp = self.t(g, img, res, 1), self.t(g, img, res, p)
            return t1 / tp if t1 and tp else None
        return self.per_image(g, res, one)

    def ratio_t1_ts(self, g, res):
        def one(img):
            ts, t1 = self.t(g, img, res), self.t(g, img, res, 1)
            return t1 / ts if ts and t1 else None
        return self.per_image(g, res, one)


def tabular(spec, header, body):
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
             " & ".join(f"\\textbf{{{h}}}" for h in header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in body]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print("  " + str(path))


def environment_table(env, tdir):
    cpu, topo, omp = env.get("cpu", ""), env.get("topology", {}), env.get("omp", {})
    caches = [f"{k}: {field(cpu, k + ' cache')}" for k in ("L1d", "L2", "L3") if field(cpu, k + " cache")]
    mem = field(env.get("memory_start", ""), "MemTotal")
    mem = f"{int(mem.split()[0]) / 1024 ** 2:.1f} GiB".replace(".", ",") if mem else None
    smt = {"1": "ativo", "0": "inativo"}.get(str(topo.get("smt_active")), "não registrado")
    system = field(env.get("os_release", ""), "PRETTY_NAME")
    kernel = (env.get("kernel") or "").split()
    kernel = f"Linux {kernel[2]}" if len(kernel) > 2 else None
    checks = env.get("affinity_checks", [])
    distinct = all(not c.get("uses_smt_siblings") for c in checks) if checks else None
    rows = [
        ("Processador", field(cpu, "Model name", "Nome do modelo")),
        ("Núcleos", f"{topo.get('physical_cores_available', '--')} físicos, "
                    f"{topo.get('logical_cpus_available', '--')} lógicos; SMT {smt}" if topo else None),
        ("Memória RAM", mem),
        ("Cache", "; ".join(caches) or None),
        ("Sistema operacional", ", ".join(x for x in (system, kernel) if x) or None),
        ("Compilador", (env.get("compiler") or "").splitlines()[0] if env.get("compiler") else None),
        ("Opções de compilação", " ".join(env.get("flags", [])) or None),
        ("Biblioteca", f"OpenCV {env['opencv']} (uma thread, OpenCL desativado)" if env.get("opencv") else None),
        ("Controle de paralelismo", ", ".join(f"{k}={omp[k]}" for k in ("OMP_DYNAMIC", "OMP_PROC_BIND",
                                     "OMP_WAIT_POLICY") if k in omp) or None),
        ("Afinidade de threads", None if distinct is None else
         "taskset; uma thread por núcleo físico em todos os graus medidos" if distinct else
         "taskset; núcleos físicos primeiro, depois threads SMT"),
        ("Governador de frequência", env.get("governor")),
    ]
    body = [[name, esc(value.replace("instances", "instâncias").replace("instance", "instância"))
             if value else "não registrado"] for name, value in rows]
    write(tdir / "ambiente.tex", tabular("lp{0.62\\textwidth}", ["Componente", "Especificação"], body))


def tables(d, env, rows_all, out):
    tdir = out / "tabelas"
    environment_table(env, tdir)

    #Registro da coleta
    body = [
        ["Situação final", str(env.get("status", "--")).replace("_", "\\_")],
        ["Modo", str(env.get("mode", "--")).replace("_", "\\_")],
        ["Início", str(env.get("start", "--"))[:16].replace("T", " ")],
        ["Término", str(env.get("finish", "--"))[:16].replace("T", " ")],
        ["Configurações", str(len({k for k in d.stats}))],
        ["Medições válidas", str(sum(s["n"] for s in d.stats.values()))],
        ["Configurações com CV acima de 5\\%",
         str(sum(1 for s in d.stats.values() if s["cv"] > CV_LIMIT))],
        ["Semente de aleatorização", str(env.get("random_seed", "--"))],
    ]
    write(tdir / "registro_coleta.tex", tabular("ll", ["Item", "Valor"], body))

    #Dispersao por resolucao
    body = []
    for res in d.resolutions:
        cvs = [s["cv"] for k, s in d.stats.items() if k[2] == res]
        body.append([RESOLUTIONS.get(res, res), str(len(cvs)), br(statistics.median(cvs)),
                     br(max(cvs)), str(sum(1 for c in cvs if c > CV_LIMIT))])
    write(tdir / "dispersao.tex", tabular("lrrrr", ["Resolução", "Configurações",
          "CV mediano (\\%)", "CV máximo (\\%)", "CV $>$ 5\\%"], body))

    #Speedup por grupo
    header = ["Resolução", "$T_s$ (ms)", "$T_1/T_s$"] + [f"$S({p})$" for p in THREADS[1:]]
    for g in d.groups:
        body = [[RESOLUTIONS.get(res, res), br(d.ts(g, res), 3), br(d.ratio_t1_ts(g, res))]
                + [br(d.speedup(g, res, p)) for p in THREADS[1:]] for res in d.resolutions]
        write(tdir / f"speedup_{g}.tex", tabular("lrrrrrr", header, body))

    #Efeito das refatoracoes
    cols = [r for r in ("base", "uhd4k", "uhd8k") if r in d.resolutions]
    body = [[GROUPS.get(g, g), REFACTORINGS.get(g, "--")] + [br(d.ratio_t1_ts(g, r)) for r in cols]
            for g in d.groups]
    write(tdir / "refatoracoes.tex", tabular("p{0.2\\textwidth}p{0.3\\textwidth}" + "r" * len(cols),
          ["Grupo", "Refatoração"] + [RESOLUTIONS[r] for r in cols], body))

    #Escalabilidade da versao paralelizada (T_1/T_p) em 4K
    if "uhd4k" in d.resolutions:
        body = [[GROUPS.get(g, g)] + [br(d.internal(g, "uhd4k", p)) for p in THREADS[1:]] for g in d.groups]
        write(tdir / "escalabilidade.tex", tabular("l" + "r" * len(THREADS[1:]), ["Grupo"] +
              [f"$T_1/T_{{{p}}}$" for p in THREADS[1:]], body))

    #Orcamento por quadro: menor p cuja media (das tres imagens) cabe no orcamento
    body = []
    for g in d.groups:
        for i, res in enumerate(r for r in ("uhd4k", "uhd8k") if r in d.resolutions):
            ts = d.ts(g, res)
            cells = []
            for budget in BUDGETS_MS:
                if ts is not None and ts <= budget:
                    cells.append("seq.")
                    continue
                fits = [p for p in THREADS if (d.tp(g, res, p) or float("inf")) <= budget]
                cells.append(str(fits[0]) if fits else "não cabe")
            body.append([GROUPS.get(g, g) if i == 0 else "", RESOLUTIONS[res], br(ts, 3)] + cells)
    write(tdir / "orcamento.tex", tabular("llrrr", ["Grupo", "Resolução", "$T_s$ (ms)",
          "$\\leq$ 33,3 ms", "$\\leq$ 16,7 ms"], body))

    #Escalonamento (somente se a coleta registrar o campo "schedule")
    schedules = sorted({k[5] for k in d.stats if k[3] == "paralelizada"})
    if len(schedules) > 1:
        body = []
        for g in d.groups:
            for res in d.resolutions:
                for p in THREADS:
                    cells = []
                    for sch in ("static", "dynamic", "guided"):
                        vals = [d.stats.get((g, img, res, "paralelizada", p, sch)) for img in d.images]
                        vals = [v for v in vals if v]
                        cells.append(f"{br(statistics.mean(v['mean'] for v in vals), 3)} "
                                     f"({br(statistics.mean(v['cv'] for v in vals), 1)})" if vals else "--")
                    if any(c != "--" for c in cells[1:]):
                        body.append([RESOLUTIONS.get(res, res), str(p)] + cells)
        write(tdir / "escalonamento.tex", tabular("llrrr", ["Resolução", "$p$", "\\textit{static}",
              "\\textit{dynamic}", "\\textit{guided}"], body))

    #Apendice: media e CV por imagem
    header = ["Imagem", "Resolução", "$T_s$"] + [f"$T_{{{p}}}$" for p in THREADS]
    for g in d.groups:
        body = []
        for img in d.images:
            for j, res in enumerate(d.resolutions):
                cells = []
                for p in (None,) + THREADS:
                    key = (g, img, res, "sequencial", 1, "static") if p is None else \
                          (g, img, res, "paralelizada", p, "static")
                    s = d.stats.get(key)
                    cells.append(f"{br(s['mean'], 3)} $\\pm$ {br(s['sd'], 3)}" if s else "--")
                body.append([IMAGES.get(img, img) if j == 0 else "", RESOLUTIONS.get(res, res)] + cells)
        write(tdir / f"apendice_{g}.tex", tabular("ll" + "r" * (1 + len(THREADS)), header, body))


def charts(d, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ImportError:
        print("AVISO: matplotlib ausente; graficos nao gerados (pip install matplotlib)")
        return
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
        "lines.linewidth": 1.6, "lines.markersize": 6, "pdf.fonttype": 42,
    })
    comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
    gdir = out / "graficos"
    gdir.mkdir(parents=True, exist_ok=True)

    def color(g):
        i = list(GROUPS).index(g) if g in GROUPS else len(GROUPS) - 1
        return GROUP_COLORS[i % len(GROUP_COLORS)], GROUP_MARKERS[i % len(GROUP_MARKERS)]

    def ideal(ax):
        ax.plot(THREADS, THREADS, linestyle="--", color=MUTED, linewidth=1, label="Ideal ($S=p$)")

    def finish(fig, ax, name, ylabel, xlabel="Número de threads ($p$)"):
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.yaxis.set_major_formatter(comma)
        ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
        fig.tight_layout()
        fig.savefig(gdir / name)
        plt.close(fig)
        print("  " + str(gdir / name))

    #S(p) por grupo, uma curva por resolucao
    for g in d.groups:
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        ideal(ax)
        for i, res in enumerate(d.resolutions):
            ys = [d.speedup(g, res, p) for p in THREADS]
            ax.plot(THREADS, ys, color=RES_COLORS[i], marker=RES_MARKERS[i], label=RESOLUTIONS.get(res, res))
        ax.set_xticks(THREADS)
        finish(fig, ax, f"speedup_{g}.pdf", "Speedup $S(p)$")

    #S(p) em 4K, todos os grupos
    if "uhd4k" in d.resolutions:
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        ideal(ax)
        for g in d.groups:
            c, m = color(g)
            ax.plot(THREADS, [d.speedup(g, "uhd4k", p) for p in THREADS], color=c, marker=m,
                    label=GROUPS.get(g, g))
        ax.set_xticks(THREADS)
        finish(fig, ax, "speedup_4k.pdf", "Speedup $S(p)$")

    #T_1/T_p em 4K, todos os grupos
    if "uhd4k" in d.resolutions:
        fig, ax = plt.subplots(figsize=(6.3, 3.4))
        ideal(ax)
        for g in d.groups:
            c, m = color(g)
            ax.plot(THREADS, [d.internal(g, "uhd4k", p) for p in THREADS], color=c, marker=m,
                    label=GROUPS.get(g, g))
        ax.set_xticks(THREADS)
        finish(fig, ax, "escalabilidade_4k.pdf", "$T_1/T_p$")

    # E(8) por resolucao, todos os grupos
    pmax = max(p for p in THREADS if any(d.speedup(g, r, p) for g in d.groups for r in d.resolutions))
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    xs = list(range(len(d.resolutions)))
    for g in d.groups:
        c, m = color(g)
        ys = [(d.speedup(g, r, pmax) or float("nan")) / pmax * 100 for r in d.resolutions]
        ax.plot(xs, ys, color=c, marker=m, label=GROUPS.get(g, g))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{RESOLUTIONS.get(r, r)}\n{str(MEGAPIXELS.get(r, '')).replace('.', ',')} Mpx"
                       for r in d.resolutions])
    ax.set_ylim(bottom=0)
    finish(fig, ax, "eficiencia_resolucao.pdf", f"Eficiência $E({pmax})$ (%)", xlabel="Resolução")

    #tempo em 4K com os orcamentos por quadro (escala log)
    if "uhd4k" in d.resolutions:
        fig, ax = plt.subplots(figsize=(6.3, 3.6))
        labels = ["seq."] + [str(p) for p in THREADS]
        xs = list(range(len(labels)))
        for g in d.groups:
            c, m = color(g)
            ys = [d.ts(g, "uhd4k")] + [d.tp(g, "uhd4k", p) for p in THREADS]
            ax.plot(xs, [y if y else float("nan") for y in ys], color=c, marker=m, label=GROUPS.get(g, g))
        for budget, style in zip(BUDGETS_MS, (":", "-.")):
            ax.axhline(budget, color=MUTED, linestyle=style, linewidth=1)
            ax.annotate(f"{br(budget, 1)} ms", xy=(xs[-1], budget), xytext=(4, 2),
                        textcoords="offset points", color=MUTED, fontsize=8)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        finish(fig, ax, "orcamento_4k.pdf", "Tempo médio do núcleo (ms)",
               xlabel="Variante sequencial e número de threads")


def summary_csv(d, out):
    path = out / "resumo.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["grupo", "imagem", "resolucao", "variante", "threads", "politica",
                    "n", "media_ms", "desvio_ms", "cv_pct", "speedup", "eficiencia"])
        for key in sorted(d.stats):
            g, img, res, var, p, sch = key
            s = d.stats[key]
            ts = d.t(g, img, res)
            sp = ts / s["mean"] if var == "paralelizada" and ts else None
            w.writerow([g, img, res, var, p, sch, s["n"], f"{s['mean']:.6f}", f"{s['sd']:.6f}",
                        f"{s['cv']:.3f}", f"{sp:.4f}" if sp else "", f"{sp / p:.4f}" if sp else ""])
    print("  " + str(path))


def main():
    parser = argparse.ArgumentParser(description="Gera as tabelas e os graficos do Capitulo 7 a partir de uma coleta.")
    parser.add_argument("coleta", type=Path, help="pasta resultados/<data>-<codigo>")
    parser.add_argument("--saida", type=Path, help="pasta de saida (padrao: <coleta>/analise)")
    args = parser.parse_args()
    rows, env = load(args.coleta)
    out = args.saida or args.coleta / "analise"
    d = Data(aggregate(rows))
    incomplete = [k for k, s in d.stats.items() if s["n"] != 3]
    if incomplete:
        print(f"AVISO: {len(incomplete)} configuracoes sem exatamente 3 medicoes")
    print(f"{len(rows)} medicoes, {len(d.stats)} configuracoes, grupos: {', '.join(d.groups)}")
    tables(d, env, rows, out)
    summary_csv(d, out)
    charts(d, out)


if __name__ == "__main__":
    main()
