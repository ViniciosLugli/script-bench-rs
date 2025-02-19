import json
import glob
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetric:
    estimate: float
    lower_bound: float
    upper_bound: float
    unit: str

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> Optional["BenchmarkMetric"]:
        if not data or not isinstance(data, dict):
            return None

        try:
            return cls(
                estimate=float(data.get("estimate", 0.0)),
                lower_bound=float(data.get("lower_bound", 0.0)),
                upper_bound=float(data.get("upper_bound", 0.0)),
                unit=str(data.get("unit", "ns")),
            )
        except (TypeError, ValueError) as e:
            logger.debug(f"Error creating metric from data {data}: {e}")
            return None


@dataclass
class BenchmarkResult:
    id: str
    measured_values: List[float]
    iteration_count: List[int]
    typical: Optional[BenchmarkMetric]
    unit: str
    report_directory: Optional[str] = None
    mean: Optional[BenchmarkMetric] = None
    median: Optional[BenchmarkMetric] = None
    median_abs_dev: Optional[BenchmarkMetric] = None
    slope: Optional[BenchmarkMetric] = None
    throughput: List[Any] = None
    change: Optional[Any] = None

    @property
    def statistics(self) -> dict:
        if not self.measured_values:
            return {}

        values_ms = [v / 1e6 for v in self.measured_values]
        q1, q3 = np.percentile(values_ms, [25, 75])
        iqr = q3 - q1
        return {
            "iterations": {
                "count": len(self.iteration_count),
                "min": min(self.iteration_count, default=0),
                "max": max(self.iteration_count, default=0),
            },
            "measurements": {
                "min": min(values_ms, default=0),
                "max": max(values_ms, default=0),
                "mean": np.mean(values_ms) if values_ms else 0,
                "std": np.std(values_ms) if values_ms else 0,
                "median": np.median(values_ms) if values_ms else 0,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
            },
            "typical": {
                "estimate": (self.typical.estimate / 1e6) if self.typical else 0,
                "confidence_interval": [
                    (self.typical.lower_bound / 1e6) if self.typical else 0,
                    (self.typical.upper_bound / 1e6) if self.typical else 0,
                ],
            },
        }


class BenchmarkRunner:
    def __init__(self, output_dir: Path = Path("benchmark_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.benches: Dict[str, Dict[str, BenchmarkResult]] = {}
        self.unique_ids = set()
        sns.set()
        self.language_colors = {}
        self.language_names = set()

    def compile_wasm(self) -> bool:
        logger.info("Starting WASM compilation")
        start_time = time.time()

        try:
            rustup_check = subprocess.run(
                "rustup target list --installed",
                shell=True,
                capture_output=True,
                text=True,
            )

            if "wasm32-unknown-unknown" not in rustup_check.stdout:
                logger.info("WASM target not found, installing...")
                subprocess.run(
                    "rustup target add wasm32-unknown-unknown",
                    shell=True,
                    check=True,
                    capture_output=True,
                )

            scripts_dir = Path("scripts")
            if not scripts_dir.exists():
                logger.warning("Scripts directory not found, creating...")
                scripts_dir.mkdir(exist_ok=True)

            wasm_source = scripts_dir / "sort_userdata.wasm.rs"
            wasm_output = scripts_dir / "sort_userdata.wasm"

            if not wasm_source.exists():
                logger.error(f"WASM source file not found: {wasm_source}")
                return False

            logger.info(f"Compiling WASM: {wasm_source} -> {wasm_output}")

            result = subprocess.run(
                f"rustc --target wasm32-unknown-unknown -Cpanic=abort -O "
                f"--crate-name sort_userdata {wasm_source} -o {wasm_output}",
                shell=True,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"WASM compilation failed: {result.stderr}")
                return False

            duration = time.time() - start_time
            logger.info(f"WASM compilation completed in {duration:.2f} seconds")

            if not wasm_output.exists():
                logger.error("WASM file was not created despite successful compilation")
                return False

            logger.info(f"WASM file created successfully: {wasm_output}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Error during WASM compilation: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during WASM compilation: {e}")
            return False

    def _parse_benchmark_result(self, data: dict) -> Optional[BenchmarkResult]:
        try:
            if "id" not in data:
                logger.warning("Benchmark data missing 'id' field")
                return None

            measured_values = data.get("measured_values", [])
            iteration_count = data.get("iteration_count", [])

            if not measured_values or not iteration_count:
                logger.warning(f"Benchmark {data['id']} missing measurement data")
                return None

            return BenchmarkResult(
                id=data["id"],
                measured_values=measured_values,
                iteration_count=iteration_count,
                unit=data.get("unit", "ns"),
                report_directory=data.get("report_directory"),
                typical=BenchmarkMetric.from_dict(data.get("typical")),
                mean=BenchmarkMetric.from_dict(data.get("mean")),
                median=BenchmarkMetric.from_dict(data.get("median")),
                median_abs_dev=BenchmarkMetric.from_dict(data.get("median_abs_dev")),
                slope=BenchmarkMetric.from_dict(data.get("slope")),
                throughput=data.get("throughput", []),
                change=data.get("change"),
            )
        except Exception as e:
            logger.error(f"Error parsing benchmark result: {e}")
            logger.debug(f"Problematic data: {data}")
            return None

    def run_single_benchmark(self, name: str) -> List[BenchmarkResult]:
        logger.info(f"Starting benchmark: {name}")
        start_time = time.time()

        try:
            proc = subprocess.run(
                f"cargo criterion --bench {name} --message-format json --features {name}",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )

            results = []
            valid_lines = 0
            total_lines = 0

            for line in proc.stdout.splitlines():
                total_lines += 1
                try:
                    data = json.loads(line)

                    if data.get("reason") == "benchmark-complete":
                        valid_lines += 1
                        logger.debug(f"Processing benchmark result for {name}")

                        result = self._parse_benchmark_result(data)
                        if result:
                            results.append(result)
                            self._log_benchmark_progress(result)

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing line in {name}: {e}")
                    continue

            duration = time.time() - start_time
            logger.info(f"Benchmark {name} completed in {duration:.2f} seconds")
            logger.debug(
                f"Processed {total_lines} lines, {valid_lines} valid benchmark results"
            )

            return results

        except subprocess.CalledProcessError as e:
            logger.error(f"Benchmark {name} execution failed: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in benchmark {name}: {e}")
            return []

    def _log_benchmark_progress(self, result: BenchmarkResult):
        stats = result.statistics
        if not stats:
            logger.warning(f"No statistics available for {result.id}")
            return

        logger.info(
            f"""
Benchmark Progress - {result.id}:
- Iterations: {stats['iterations']['count']} ({stats['iterations']['min']} to {stats['iterations']['max']})
- Measurements (ms):
  * Min: {stats['measurements']['min']:.2f}
  * Max: {stats['measurements']['max']:.2f}
  * Mean: {stats['measurements']['mean']:.2f}
  * Median: {stats['measurements']['median']:.2f}
  * Std Dev: {stats['measurements']['std']:.2f}
  * Q1: {stats['measurements']['q1']:.2f}
  * Q3: {stats['measurements']['q3']:.2f}
  * IQR: {stats['measurements']['iqr']:.2f}
- Typical Estimate: {stats['typical']['estimate']:.2f}ms
  * Confidence Interval: [{stats['typical']['confidence_interval'][0]:.2f}, {stats['typical']['confidence_interval'][1]:.2f}]
"""
        )

    def run_all_benchmarks(self, bench_dir: str = "benches/*.rs") -> None:
        logger.info("Starting benchmark suite")
        start_time = time.time()

        bench_files = list(glob.glob(bench_dir))
        if not bench_files:
            logger.error(f"No benchmark files found in {bench_dir}")
            return

        logger.info(f"Found {len(bench_files)} benchmark files")
        successful_benchmarks = 0

        for bench_file in bench_files:
            name = Path(bench_file).stem
            self.language_names.add(name)
            logger.info(f"Processing benchmark file: {bench_file}")

            results = self.run_single_benchmark(name)
            if results:
                successful_benchmarks += 1
                for result in results:
                    if name not in self.benches:
                        self.benches[name] = {}
                    self.benches[name][result.id] = result
                    self.unique_ids.add(result.id)

        duration = time.time() - start_time
        logger.info(f"Benchmark suite completed in {duration:.2f} seconds")
        logger.info(
            f"Successful benchmarks: {successful_benchmarks}/{len(bench_files)}"
        )
        logger.info(f"Unique test cases: {len(self.unique_ids)}")
        self._assign_language_colors()

    def _assign_language_colors(self):
        num_languages = len(self.language_names)
        palette = sns.color_palette("husl", num_languages)
        for i, language in enumerate(sorted(self.language_names)):
            self.language_colors[language] = palette[i]

    def generate_plots(self) -> None:
        logger.info("Generating benchmark plots")

        for benchmark_id in self.unique_ids:
            self._create_comparison_plot(benchmark_id)
            self._create_distribution_plot(benchmark_id)

    def _create_comparison_plot(self, benchmark_id: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        sorted_benches = sorted(
            self.benches.items(), key=lambda x: x[1][benchmark_id].typical.estimate
        )

        names = []
        estimates = []
        err_ranges = []
        colors = []

        for name, bench in sorted_benches:
            result = bench[benchmark_id]
            names.append(name)
            estimates.append(result.typical.estimate / 1e6)
            err_ranges.append(
                [
                    (result.typical.estimate - result.typical.lower_bound) / 1e6,
                    (result.typical.upper_bound - result.typical.estimate) / 1e6,
                ]
            )
            colors.append(self.language_colors[name])

        err_ranges = np.array(err_ranges).T

        bars = ax.bar(names, estimates, width=0.6, color=colors)
        ax.errorbar(
            names, estimates, yerr=err_ranges, fmt="none", color="black", capsize=5
        )

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}ms",
                ha="center",
                va="bottom",
            )

        ax.set_title(f"Benchmark Comparison: {benchmark_id}")
        ax.set_ylabel("Time (ms)")
        fig.autofmt_xdate()

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=self.language_colors[language])
            for language in sorted(self.language_names)
        ]
        labels = sorted(self.language_names)

        plt.savefig(
            self.output_dir / f"{benchmark_id}_comparison.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_distribution_plot(self, benchmark_id: str) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))

        data = []
        names = []
        colors = []

        for name, bench in self.benches.items():
            result = bench[benchmark_id]
            values = [v / 1e6 for v in result.measured_values]
            data.append(values)
            names.append(name)
            colors.append(self.language_colors[name])

        parts = ax.violinplot(data, showmeans=True, showmedians=True)

        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[i])
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names)

        ax.set_title(f"Measurement Distribution: {benchmark_id}")
        ax.set_ylabel("Time (ms)")

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=self.language_colors[language])
            for language in sorted(self.language_names)
        ]
        labels = sorted(self.language_names)

        plt.savefig(
            self.output_dir / f"{benchmark_id}_distribution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def generate_report(self) -> None:
        report_path = self.output_dir / "benchmark_report.md"

        with open(report_path, "w") as f:
            f.write("# Benchmark Report\n\n")

            for benchmark_id in self.unique_ids:
                f.write(f"## {benchmark_id}\n\n")

                for name, bench in self.benches.items():
                    result = bench[benchmark_id]
                    stats = result.statistics

                    f.write(f"### {name}\n\n")
                    f.write("#### Summary Statistics\n")
                    f.write(f"- Total Iterations: {stats['iterations']['count']}\n")
                    f.write(
                        f"- Measurement Range: {stats['measurements']['min']:.2f}ms to {stats['measurements']['max']:.2f}ms\n"
                    )
                    f.write(
                        f"- Mean Time: {stats['measurements']['mean']:.2f}ms (±{stats['measurements']['std']:.2f}ms)\n"
                    )
                    f.write(f"- Median Time: {stats['measurements']['median']:.2f}ms\n")
                    f.write(
                        f"- Q1: {stats['measurements']['q1']:.2f}ms, Q3: {stats['measurements']['q3']:.2f}ms, IQR: {stats['measurements']['iqr']:.2f}ms\n"
                    )
                    f.write(
                        f"- Typical Estimate: {stats['typical']['estimate']:.2f}ms\n"
                    )
                    f.write(
                        f"- 95% Confidence Interval: [{stats['typical']['confidence_interval'][0]:.2f}ms, {stats['typical']['confidence_interval'][1]:.2f}ms]\n\n"
                    )

        logger.info(f"Generated benchmark report: {report_path}")


def main():
    try:
        runner = BenchmarkRunner()

        if not runner.compile_wasm():
            logger.error("WASM compilation failed, stopping benchmark suite")
            return

        runner.run_all_benchmarks()

        if runner.benches:
            runner.generate_plots()
            runner.generate_report()
        else:
            logger.error("No valid benchmark results to process")

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
