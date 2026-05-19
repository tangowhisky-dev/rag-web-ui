"""
CLI entry point for retrieval auto-tuning.

Usage:
    python -m app.cli.tune --iterations 20 --state ingest_state.json
    python -m app.cli.tune --iterations 10 --seed 42
"""

import argparse
import json
import sys
import os

from app.services.auto_tune import (
    run_tuning_loop,
    save_best_config,
    load_best_config,
    TuningResult,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-tune retrieval config via eval-driven search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m app.cli.tune --iterations 20 --state ingest_state.json
    python -m app.cli.tune --iterations 10 --seed 42 --output ./best.json
    python -m app.cli.tune --load  # Load best config and print
""",
    )
    parser.add_argument(
        "--iterations", type=int, default=20,
        help="Maximum configs to evaluate (default: 20)",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Stop if F1 doesn't improve for this many iterations (default: 5)",
    )
    parser.add_argument(
        "--state", default="ingest_state.json",
        help="Path to ingest_state.json with kb_id and questions",
    )
    parser.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "http://localhost:8000/api"),
        help="API base URL (default: from BASE_URL env or localhost:8000/api)",
    )
    parser.add_argument(
        "--username", default=os.getenv("USERNAME", "tune_user"),
        help="Eval username",
    )
    parser.add_argument(
        "--password", default=os.getenv("PASSWORD", "tune_pass"),
        help="Eval password",
    )
    parser.add_argument(
        "--email", default=os.getenv("EMAIL", "tune@example.com"),
        help="Eval email",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output path for best config (default: .gsd/tuning/best_config.json)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--load", action="store_true",
        help="Load best config from disk and print (no tuning)",
    )

    args = parser.parse_args()

    if args.load:
        config = load_best_config(args.output)
        if config:
            print(json.dumps(config, indent=2))
        else:
            print("No best config found.", file=sys.stderr)
            sys.exit(1)
        return

    # Load state
    if not os.path.exists(args.state):
        print(f"State file not found: {args.state}", file=sys.stderr)
        sys.exit(1)

    with open(args.state) as f:
        state = json.load(f)

    kb_id = state["kb_id"]
    questions = state["questions"]
    print(f"Loaded state: KB id={kb_id}, {len(questions)} questions")
    print(f"Tuning: iterations={args.iterations}, patience={args.patience}")
    print(f"API: {args.base_url}")
    print()

    # Run tuning
    result = run_tuning_loop(
        questions=questions,
        kb_id=kb_id,
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        email=args.email,
        max_iterations=args.iterations,
        patience=args.patience,
        seed=args.seed,
    )

    # Save
    output_path = save_best_config(result, kb_id, args.output)

    # Summary
    print()
    print("=" * 60)
    print("Tuning complete!")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Converged:  {result.converged_at or 'No'}")
    print(f"  Best F1:    {result.best_result.mean_f1:.4f}")
    print(f"  Best EM:    {result.best_result.mean_em:.4f}")
    print(f"  Best Hit:   {result.best_result.hit_rate:.4f}")
    print(f"  Best Lat:   {result.best_result.mean_latency_ms:.0f}ms")
    print(f"  Config:     {result.best_config.to_dict()}")
    print(f"  Saved to:   {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
