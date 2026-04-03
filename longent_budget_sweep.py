#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import secrets
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

import fiveway_extreme20 as fx
import model_gateway_tester as mgt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep max_output_tokens for two Longent models on one shared fresh extreme test set."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-tests", type=int, default=30)
    parser.add_argument("--longent-key", type=str, required=True)
    parser.add_argument(
        "--longent-url",
        type=str,
        default="https://longent.tech/v1/responses",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="gpt-5.4-fast(xhigh)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="gpt-5.4(xhigh)",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="xhigh",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 9192],
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd()
        / f"longent_budget_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument(
        "--prompts-output",
        type=pathlib.Path,
        default=None,
    )
    return parser.parse_args()


def choose_winner(summary: dict[str, Any], provider_names: list[str]) -> str:
    ranked = []
    for provider in provider_names:
        stats = summary[provider]
        ranked.append(
            {
                "provider": provider,
                "accuracy": stats["accuracy_scored_all_pct"],
                "completion": stats["completion_rate_pct"],
                "latency": stats["avg_elapsed_ms"],
            }
        )
    ranked.sort(
        key=lambda item: (
            -1 if item["accuracy"] is None else -item["accuracy"],
            -1 if item["completion"] is None else -item["completion"],
            float("inf") if item["latency"] is None else item["latency"],
            item["provider"],
        )
    )
    if len(ranked) >= 2:
        first = ranked[0]
        second = ranked[1]
        if (
            first["accuracy"] == second["accuracy"]
            and first["completion"] == second["completion"]
            and first["latency"] == second["latency"]
        ):
            return "tie"
    return ranked[0]["provider"]


def build_combined_report(
    *,
    seed: int,
    total_tests: int,
    budgets: list[int],
    selection_meta: dict[str, Any],
    prompts_output: pathlib.Path,
    prompt_rows: list[dict[str, Any]],
    per_budget: list[dict[str, Any]],
) -> dict[str, Any]:
    winner_counts: dict[str, int] = {}
    for item in per_budget:
        winner = item["winner"]
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    return {
        "meta": {
            "seed": seed,
            "total_tests": total_tests,
            "budgets": budgets,
            "selection": selection_meta,
            "prompts_output": str(prompts_output),
            "note": "Each budget uses the same fresh extreme test set to isolate the effect of max_output_tokens.",
        },
        "winner_counts": winner_counts,
        "per_budget": per_budget,
        "prompts": prompt_rows,
    }


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    tests, selection_meta = fx.build_extreme_suite(seed, args.total_tests)

    prompts_output = args.prompts_output
    if prompts_output is None:
        prompts_output = args.output.with_name(args.output.stem + "_prompts.json")

    prompt_rows = [
        {
            "test_id": test.test_id,
            "family": test.family,
            "prompt": test.prompt,
        }
        for test in tests
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts_output.parent.mkdir(parents=True, exist_ok=True)
    prompts_output.write_text(json.dumps(prompt_rows, ensure_ascii=False, indent=2))

    per_budget: list[dict[str, Any]] = []
    checkpoint_path = args.output.with_name(args.output.stem + "_checkpoint.json")

    for budget in args.budgets:
        providers = [
            mgt.ProviderConfig(
                name=f"Longent {args.model_a} @ {budget}",
                url=args.longent_url,
                api_key=args.longent_key,
                model=args.model_a,
                reasoning_mode="none",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=budget,
                needs_user_agent=True,
            ),
            mgt.ProviderConfig(
                name=f"Longent {args.model_b} @ {budget}",
                url=args.longent_url,
                api_key=args.longent_key,
                model=args.model_b,
                reasoning_mode="none",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=budget,
                needs_user_agent=True,
            ),
        ]
        rows: list[dict[str, Any]] = []
        total_requests = len(tests) * len(providers)
        completed_requests = 0
        for test_index, test in enumerate(tests, start=1):
            print(
                f"[budget {budget}] [test {test_index}/{len(tests)}] dispatching {len(providers)} providers",
                file=sys.stderr,
                flush=True,
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(providers)
            ) as executor:
                future_map = {
                    executor.submit(mgt.run_request, provider, test.prompt): provider
                    for provider in providers
                }
                completed_for_test: dict[str, dict[str, Any]] = {}
                for future in concurrent.futures.as_completed(future_map):
                    provider = future_map[future]
                    completed_requests += 1
                    row = future.result()
                    row.update(
                        {
                            "test": test.test_id,
                            "family": test.family,
                            "provider": provider.name,
                            "expected": test.expected,
                        }
                    )
                    row["correct"] = row.get("output_text") == test.expected
                    completed_for_test[provider.name] = row
                    print(
                        f"[budget {budget}] [{completed_requests}/{total_requests}] {provider.name} :: {test.test_id}",
                        file=sys.stderr,
                        flush=True,
                    )
            for provider in providers:
                rows.append(completed_for_test[provider.name])

        provider_names = [provider.name for provider in providers]
        summary = fx.summarize_rows(rows, provider_names)
        winner = choose_winner(summary, provider_names)
        report_rows = mgt.build_report_rows(tests, rows, include_prompt_text=False)
        budget_report = {
            "budget": budget,
            "providers": [
                asdict(provider) | {"api_key": "***redacted***"} for provider in providers
            ],
            "summary": summary,
            "winner": winner,
            "rows": report_rows,
        }
        per_budget.append(budget_report)
        checkpoint = build_combined_report(
            seed=seed,
            total_tests=args.total_tests,
            budgets=args.budgets,
            selection_meta=selection_meta,
            prompts_output=prompts_output,
            prompt_rows=prompt_rows,
            per_budget=per_budget,
        )
        checkpoint_path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2))

    final_report = build_combined_report(
        seed=seed,
        total_tests=args.total_tests,
        budgets=args.budgets,
        selection_meta=selection_meta,
        prompts_output=prompts_output,
        prompt_rows=prompt_rows,
        per_budget=per_budget,
    )
    args.output.write_text(json.dumps(final_report, ensure_ascii=False, indent=2))
    checkpoint_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_output": str(checkpoint_path),
                "winner_counts": final_report["winner_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
