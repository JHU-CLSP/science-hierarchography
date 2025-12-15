#!/usr/bin/env python3

import argparse
import sys
import os
from pathlib import Path
import csv
from datetime import datetime
import random
from evaluate import get_evaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate paper hierarchies using search queries")

    parser.add_argument("--hierarchy", type=str, required=True,
                       help="Path to the hierarchy JSON file to evaluate")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to the folder containing paper files (TXT or JSON)")

    parser.add_argument("--output", type=str, default="search_evaluation_results.csv",
                       help="Path to the output CSV file (default: search_evaluation_results.csv)")
    parser.add_argument("--detailed_output", type=str, default="evaluation_details.csv",
                       help="Path to the per-sample detailed CSV (default: evaluation_details.csv)")
    parser.add_argument("--error_output", type=str, default="error_cases.jsonl",
                       help="Path to structured error cases JSONL (default: error_cases.jsonl)")
    parser.add_argument("--evaluator", type=str, default="llama",
                       help="Evaluator type to use: llama, qwen, gemma (default: llama)")
    parser.add_argument("--test_count", type=int, default=50,
                       help="Number of papers to test (default: 50)")
    parser.add_argument("--embedding_key", type=str, default=None,
                       help="Embedding key to use: problem, solution, results (default: None)")
    parser.add_argument("--query_type", type=str, default="random",
                       help="Query type: problem, challenge, solution, result, combined, random (default: random)")
    parser.add_argument("--evaluate_type", type=str, default="normal",
                       help="Evaluation type: normal, question (default: normal)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--run_time", type=int, default=1,
                       help="Number of times to run the evaluation on the hierarchy (default: 1)")

    args = parser.parse_args()

    hierarchy_path = Path(args.hierarchy)
    if not hierarchy_path.exists():
        print(f"Error: Hierarchy file not found: {args.hierarchy}")
        return 1

    data_path = Path(args.data)
    if not data_path.exists() or not data_path.is_dir():
        print(f"Error: Data folder not found: {args.data}")
        return 1

    random.seed(args.seed)

    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        evaluator = get_evaluator(args.evaluator,
                                  embedding_key=args.embedding_key,
                                  query_type=args.query_type,
                                  evaluate_type=args.evaluate_type)
    except ValueError as e:
        print(f"Error initializing evaluator: {e}")
        return 1

    print(f"\n{'='*60}")
    print(f"Running paper search evaluation")
    print(f"Hierarchy: {hierarchy_path.name}")
    print(f"Data folder: {data_path}")
    print(f"Test count: {args.test_count}")
    print(f"Run times: {args.run_time}")
    print(f"Output file: {output_file}")
    print(f"Evaluator: {args.evaluator}")
    print(f"Query type: {args.query_type}")
    print(f"{'='*60}\n")

    all_results = []

    for run in range(1, args.run_time + 1):
        print(f"\n{'-'*60}")
        print(f"Run {run}/{args.run_time}")
        print(f"{'-'*60}")

        try:
            eval_kwargs = dict(
                hierarchy_path=str(hierarchy_path),
                abstracts_folder=str(data_path),
                test_count=args.test_count,
                output_file=str(output_file),
                embedding_key=args.embedding_key,
                query_type=args.query_type,
                run_number=run,
            )
            import inspect
            sig = inspect.signature(evaluator.evaluate_hierarchy)
            if 'detailed_output_file' in sig.parameters:
                eval_kwargs['detailed_output_file'] = str(Path(args.detailed_output))
            if 'error_output_file' in sig.parameters:
                eval_kwargs['error_output_file'] = str(Path(args.error_output))

            results = evaluator.evaluate_hierarchy(**eval_kwargs)

            if results:
                all_results.append(results)
                print(f"\nRun {run} completed successfully.")
                print(f"Results saved to: {output_file}")

                print("\nRun Summary:")
                for level, acc in results["accuracy"].items():
                    print(f"Level {level} accuracy: {acc:.2%}")

        except Exception as e:
            print(f"Error during evaluation run {run}: {e}")
            import traceback
            traceback.print_exc()
            if run == 1 and args.run_time > 1:
                print("First run failed, aborting remaining runs.")
                return 1

    if args.run_time > 1 and all_results:
        print(f"\n{'='*60}")
        print(f"Summary of all {args.run_time} runs")
        print(f"{'='*60}")

        max_level = max(max(result["accuracy"].keys()) for result in all_results if result["accuracy"])

        avg_accuracies = {}
        level_stats = {}

        for level in range(max_level + 1):
            level_accs = [result["accuracy"].get(level, 0) for result in all_results
                          if result["accuracy"].get(level, 0) > 0]

            if level_accs:
                avg_acc = sum(level_accs) / len(level_accs)
                avg_accuracies[level] = avg_acc

                min_acc = min(level_accs)
                max_acc = max(level_accs)
                variance = sum((acc - avg_acc) ** 2 for acc in level_accs) / len(level_accs)
                std_dev = variance ** 0.5

                level_stats[level] = {
                    "avg": avg_acc,
                    "min": min_acc,
                    "max": max_acc,
                    "std_dev": std_dev,
                    "runs": len(level_accs)
                }

                print(f"Level {level} - Avg: {avg_acc:.2%}, Min: {min_acc:.2%}, Max: {max_acc:.2%}, StdDev: {std_dev:.4f}, Runs: {len(level_accs)}")

        summary_file = output_file.parent / f"{output_file.stem}_summary{output_file.suffix}"
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Level",
                "Average Accuracy",
                "Min Accuracy",
                "Max Accuracy",
                "Std Dev",
                "Run Count",
                "Hierarchy",
                "Query Type",
                "Test Count",
                "Date"
            ])

            for level, stats in level_stats.items():
                writer.writerow([
                    level,
                    f"{stats['avg']:.4f}",
                    f"{stats['min']:.4f}",
                    f"{stats['max']:.4f}",
                    f"{stats['std_dev']:.4f}",
                    stats['runs'],
                    hierarchy_path.name,
                    args.query_type,
                    args.test_count,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

        print(f"\nSummary statistics saved to: {summary_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
