import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import benchmark


class ProtocolTests(unittest.TestCase):
    def test_configuration_matrix_has_only_two_versions(self):
        configurations = benchmark.configurations(benchmark.THREADS)
        self.assertEqual(len(configurations), 270)
        self.assertEqual({row[3] for row in configurations}, {'sequencial', 'paralelizada'})
        self.assertEqual(sum(row[3] == 'sequencial' for row in configurations), 45)
        self.assertEqual(sum(row[3] == 'paralelizada' for row in configurations), 225)
        self.assertEqual(len(set(configurations)), 270)

    def hardware(self, cores=8, smt="0"):
        return {"physical_cores_available": cores, "smt_active": smt,
                "logical_cpus_available": cores * (2 if smt == "1" else 1)}

    def test_official_all_configurations(self):
        self.assertEqual(benchmark.select_threads("official", benchmark.THREADS, self.hardware()),
                         ([1, 2, 4, 6, 8], []))

    def test_official_refuses_four_cores(self):
        with self.assertRaises(ValueError):
            benchmark.select_threads("official", benchmark.THREADS, self.hardware(4))

    def test_official_refuses_active_or_unknown_smt(self):
        for value in ("1", None):
            with self.subTest(smt=value), self.assertRaises(ValueError):
                benchmark.select_threads("official", benchmark.THREADS, self.hardware(smt=value))

    def test_official_refuses_partial_threads(self):
        with self.assertRaises(ValueError):
            benchmark.select_threads("official", [1, 2, 4], self.hardware())

    def test_local_never_oversubscribes(self):
        self.assertEqual(benchmark.select_threads("local", benchmark.THREADS, self.hardware(4, "1")),
                         ([1, 2, 4], [6, 8]))

    def test_local_ht_uses_all_eight_hardware_threads(self):
        self.assertEqual(benchmark.select_threads("local-ht", benchmark.THREADS, self.hardware(4, "1")),
                         ([1, 2, 4, 6, 8], []))

    def test_local_ht_refuses_more_than_available_logical_cpus(self):
        with self.assertRaises(ValueError):
            benchmark.select_threads("local-ht", benchmark.THREADS, self.hardware(4, "0"))

    def test_verification_collects_no_times(self):
        self.assertEqual(benchmark.select_threads("verify", benchmark.THREADS, self.hardware(4, "1")),
                         ([], []))

    def test_threads_invalid(self):
        for requested in ([], [0, 2], [1, 1], [16]):
            with self.subTest(requested=requested), self.assertRaises(ValueError):
                benchmark.select_threads("local", requested, self.hardware())

    def test_three_perf_repetitions_are_not_nine(self):
        stdout = "threshold time: 0.1 seconds\n" * 3
        stderr = "Maximum resident set size (kbytes): 321\n" * 3
        self.assertEqual(benchmark.measurement(stdout, stderr, 3), ([0.1]*3, [321]*3))

    def test_places_exactly_match_pinned_cpus(self):
        environment = benchmark.thread_environment({"OMP_DYNAMIC": "false"}, [0, 3, 7, 11], 2)
        self.assertEqual(environment["OMP_PLACES"], "{0},{3}")
        self.assertEqual(environment["OMP_NUM_THREADS"], "2")
        self.assertEqual(environment["OMP_DYNAMIC"], "false")

    def test_affinity_refuses_insufficient_or_duplicate_cpus(self):
        for cpus, p in (([0, 1], 4), ([0, 0], 2)):
            with self.assertRaises(ValueError):
                benchmark.thread_environment({}, cpus, p)

    def test_scientific_and_fixed_timing(self):
        self.assertEqual(benchmark.measurement("threshold time: 1.23e-05 seconds\n",
                                              "Maximum resident set size (kbytes): 100\n", 1),
                         ([1.23e-05], [100]))

    def test_missing_or_invalid_values(self):
        for stdout, stderr in (("", ""), ("threshold time: 0 seconds\n", "Maximum resident set size (kbytes): 1\n"),
                               ("threshold time: 1 seconds\n" * 2, "Maximum resident set size (kbytes): 1\n")):
            with self.subTest(stdout=stdout), self.assertRaises(RuntimeError):
                benchmark.measurement(stdout, stderr, 1)


if __name__ == "__main__":
    unittest.main()
