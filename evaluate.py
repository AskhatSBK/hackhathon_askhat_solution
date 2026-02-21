import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text


@dataclass
class EvaluationResult:
    protocol_id: str
    accuracy_at_1: int  # 1 or 0
    recall_at_3: int  # 1 or 0
    latency_s: float
    ground_truth: str
    top_prediction: str
    top_3_predictions: list[str]
    response_json: dict


async def evaluate_single(
    client: httpx.AsyncClient,
    endpoint: str,
    json_file: Path,
    semaphore: asyncio.Semaphore,
) -> EvaluationResult:
    """Evaluate a single protocol against the endpoint."""
    async with semaphore:
        with open(json_file, "r") as f:
            data = json.load(f)

        protocol_id = data["protocol_id"]
        query = data["query"]
        ground_truth = data["gt"]
        valid_icd_codes = set(data["icd_codes"])

        if ground_truth not in valid_icd_codes:
            raise ValueError(
                f"Dataset error in {json_file.name}: gt '{ground_truth}' not in icd_codes"
            )

        start_time = time.perf_counter()
        response = await client.post(endpoint, json={"symptoms": query})
        latency_s = time.perf_counter() - start_time

        response.raise_for_status()
        result = response.json()

        diagnoses = sorted(result["diagnoses"], key=lambda x: x["rank"])
        top_3 = diagnoses[:3]

        top_prediction = diagnoses[0]["icd10_code"] if diagnoses else ""
        top_3_predictions = [d["icd10_code"] for d in top_3]

        # Accuracy@1: does the rank 1 prediction match ground truth?
        accuracy_at_1 = 1 if top_prediction == ground_truth else 0

        # Recall@3: are any of the top 3 predictions in the valid icd_codes list?
        recall_at_3 = (
            1 if any(code in valid_icd_codes for code in top_3_predictions) else 0
        )

        return EvaluationResult(
            protocol_id=protocol_id,
            accuracy_at_1=accuracy_at_1,
            recall_at_3=recall_at_3,
            latency_s=latency_s,
            ground_truth=ground_truth,
            top_prediction=top_prediction,
            top_3_predictions=top_3_predictions,
            response_json=result,
        )


async def run_evaluation(
    endpoint: str,
    dataset_dir: Path,
    parallelism: int,
) -> list[EvaluationResult]:
    """Run evaluation on all JSON files in the dataset directory."""
    console = Console()

    json_files = list(dataset_dir.glob("*.json"))
    if not json_files:
        console.print(f"[red]No JSON files found in {dataset_dir}[/red]")
        return []

    console.print(
        Panel(
            f"[bold cyan]Diagnostic Accuracy Evaluation[/bold cyan]\n\n"
            f"Endpoint: [yellow]{endpoint}[/yellow]\n"
            f"Dataset: [yellow]{dataset_dir}[/yellow]\n"
            f"Files: [yellow]{len(json_files)}[/yellow]\n"
            f"Parallelism: [yellow]{parallelism}[/yellow]",
            title="[bold white]Configuration[/bold white]",
            border_style="cyan",
        )
    )

    semaphore = asyncio.Semaphore(parallelism)
    results: list[EvaluationResult] = []
    errors: list[tuple[Path, Exception]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "[cyan]Evaluating protocols...", total=len(json_files)
            )

            async def process_file(json_file: Path):
                try:
                    result = await evaluate_single(
                        client, endpoint, json_file, semaphore
                    )
                    results.append(result)
                except Exception as e:
                    errors.append((json_file, e))
                finally:
                    progress.advance(task)

            await asyncio.gather(*[process_file(f) for f in json_files])

    if errors:
        console.print(
            f"\n[red]Encountered {len(errors)} errors during evaluation[/red]"
        )
        for path, err in errors[:5]:
            console.print(f"  [dim]• {path.name}: {err}[/dim]")
        if len(errors) > 5:
            console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")

    return results


def compute_metrics(results: list[EvaluationResult]) -> dict:
    """Compute aggregated metrics from evaluation results."""
    if not results:
        return {}

    total = len(results)
    accuracy_at_1 = sum(r.accuracy_at_1 for r in results) / total * 100
    recall_at_3 = sum(r.recall_at_3 for r in results) / total * 100
    latencies = [r.latency_s for r in results]
    avg_latency = statistics.mean(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    p50_latency = statistics.median(latencies)

    if total >= 4:
        quantiles = statistics.quantiles(latencies, n=20)
        p95_latency = quantiles[-1]
    else:
        p95_latency = max_latency

    return {
        "total_protocols": total,
        "accuracy_at_1_percent": round(accuracy_at_1, 2),
        "recall_at_3_percent": round(recall_at_3, 2),
        "latency_avg_s": round(avg_latency, 3),
        "latency_min_s": round(min_latency, 3),
        "latency_max_s": round(max_latency, 3),
        "latency_p50_s": round(p50_latency, 3),
        "latency_p95_s": round(p95_latency, 3),
    }


def write_jsonl(results: list[EvaluationResult], output_path: Path):
    """Write results to JSONL file."""
    with open(output_path, "w") as f:
        for r in results:
            line = {
                "protocol_id": r.protocol_id,
                "response": r.response_json,
                "scores": {
                    "accuracy_at_1": r.accuracy_at_1,
                    "recall_at_3": r.recall_at_3,
                    "latency_s": round(r.latency_s, 3),
                    "ground_truth": r.ground_truth,
                    "top_prediction": r.top_prediction,
                    "top_3_predictions": r.top_3_predictions,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_metrics_json(submission_name: str, metrics: dict, output_path: Path):
    """Write aggregated metrics to JSON file."""
    output_data = {
        "submission_name": submission_name,
        **metrics,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)


def display_summary(
    results: list[EvaluationResult],
    metrics: dict,
    output_jsonl: Path,
    output_json: Path,
    console: Console,
):
    """Display a beautiful summary of the evaluation results."""
    if not results:
        console.print("[red]No results to display[/red]")
        return

    # Metrics table
    metrics_table = Table(
        title="[bold]Evaluation Metrics[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )
    metrics_table.add_column("Metric", style="cyan", width=20)
    metrics_table.add_column("Value", style="green", justify="right", width=15)

    metrics_table.add_row("Accuracy@1", f"{metrics['accuracy_at_1_percent']:.2f}%")
    metrics_table.add_row("Recall@3", f"{metrics['recall_at_3_percent']:.2f}%")
    metrics_table.add_row(
        "Total Protocols", f"[bold white]{metrics['total_protocols']}[/bold white]"
    )

    # Latency table
    latency_table = Table(
        title="[bold]Latency Statistics[/bold]",
        show_header=True,
        header_style="bold magenta",
        border_style="cyan",
    )
    latency_table.add_column("Statistic", style="cyan", width=20)
    latency_table.add_column("Value (s)", style="green", justify="right", width=15)

    latency_table.add_row("Average", f"{metrics['latency_avg_s']:.3f}")
    latency_table.add_row("Min", f"{metrics['latency_min_s']:.3f}")
    latency_table.add_row("Max", f"{metrics['latency_max_s']:.3f}")
    latency_table.add_row("P50 (Median)", f"{metrics['latency_p50_s']:.3f}")
    latency_table.add_row("P95", f"{metrics['latency_p95_s']:.3f}")

    console.print()
    console.print(metrics_table)
    console.print()
    console.print(latency_table)
    console.print()

    success_text = Text()
    success_text.append("✓ ", style="bold green")
    success_text.append("Results saved to:\n", style="white")
    success_text.append(f"  JSONL:   {output_jsonl}\n", style="bold cyan")
    success_text.append(f"  Metrics: {output_json}", style="bold cyan")
    console.print(Panel(success_text, border_style="green"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostic accuracy against an endpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --endpoint http://localhost:8000/diagnose --dataset-dir ./data --name my_submission
  python main.py -e http://api.example.com/diagnose -d ./protocols -n team_alpha -p 10
        """,
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        help="Submission/project name (used for output file naming)",
    )
    parser.add_argument(
        "-e",
        "--endpoint",
        required=True,
        help="URL of the diagnostic endpoint",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        required=True,
        type=Path,
        help="Directory containing JSON protocol files",
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        default=2,
        help="Number of simultaneous requests (default: 2)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/evals"),
        help="Output directory for results (default: data/evals)",
    )

    args = parser.parse_args()
    console = Console()

    if not args.dataset_dir.exists():
        console.print(
            f"[red]Error: Dataset directory '{args.dataset_dir}' does not exist[/red]"
        )
        return 1

    if not args.dataset_dir.is_dir():
        console.print(f"[red]Error: '{args.dataset_dir}' is not a directory[/red]")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(
        run_evaluation(
            endpoint=args.endpoint,
            dataset_dir=args.dataset_dir,
            parallelism=args.parallelism,
        )
    )

    if results:
        output_jsonl = args.output_dir / f"{args.name}.jsonl"
        output_json = args.output_dir / f"{args.name}_metrics.json"

        write_jsonl(results, output_jsonl)
        metrics = compute_metrics(results)
        write_metrics_json(args.name, metrics, output_json)
        display_summary(results, metrics, output_jsonl, output_json, console)

    return 0


if __name__ == "__main__":
    exit(main())
