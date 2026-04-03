#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import pathlib
import random
import secrets
import sys
from dataclasses import asdict
from datetime import datetime
from typing import Any

import model_gateway_tester as mgt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one shared extreme-20 benchmark across four gateway routes and an optional manual self-entry."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--total-tests",
        type=int,
        default=20,
        help="How many extreme built-in tests to generate for the shared benchmark.",
    )
    parser.add_argument("--openrouter-key", type=str, required=True)
    parser.add_argument("--longent-key", type=str, required=True)
    parser.add_argument(
        "--openrouter-primary-model",
        type=str,
        default="openai/gpt-5.4",
    )
    parser.add_argument(
        "--openrouter-secondary-model",
        type=str,
        default="qwen/qwen3.6-plus:free",
    )
    parser.add_argument(
        "--longent-primary-model",
        type=str,
        default="gpt-5.4(xhigh)",
    )
    parser.add_argument(
        "--longent-secondary-model",
        type=str,
        default="gpt-5.4-fast(xhigh)",
    )
    parser.add_argument(
        "--openrouter-url",
        type=str,
        default="https://openrouter.ai/api/v1/responses",
    )
    parser.add_argument(
        "--longent-url",
        type=str,
        default="https://longent.tech/v1/responses",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--openrouter-primary-max-output-tokens",
        type=int,
        default=None,
        help="Optional override for the primary OpenRouter model only.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="xhigh",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd()
        / f"fiveway_extreme20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument(
        "--prompts-output",
        type=pathlib.Path,
        default=None,
        help="Optional prompt-only export without expected answers.",
    )
    parser.add_argument(
        "--self-answers-file",
        type=pathlib.Path,
        default=None,
        help="Optional JSON file mapping test_id to a manual answer, or list rows with test_id/output_text.",
    )
    parser.add_argument(
        "--self-provider-name",
        type=str,
        default="Codex (this session)",
    )
    return parser.parse_args()


def build_extreme_suite(seed: int, total_tests: int) -> tuple[list[mgt.TestCase], dict[str, Any]]:
    if total_tests <= 0:
        raise ValueError("--total-tests must be > 0")

    families_per_round = mgt.difficulty_family_count("extreme")
    tasks_per_family = (total_tests + families_per_round - 1) // families_per_round
    generated = mgt.build_dynamic_test_suite(seed, tasks_per_family, "extreme")
    expected_generated = tasks_per_family * families_per_round
    if len(generated) != expected_generated:
        raise ValueError(
            f"Expected {expected_generated} generated tests for extreme/{tasks_per_family}, got {len(generated)}"
        )

    selected = list(generated)
    dropped_tests: list[mgt.TestCase] = []
    if len(selected) > total_tests:
        drop_rng = random.Random(seed ^ 0x6A09E667F3BCC909)
        keep_indexes = sorted(drop_rng.sample(range(len(selected)), total_tests))
        keep_index_set = set(keep_indexes)
        dropped_tests = [test for idx, test in enumerate(selected) if idx not in keep_index_set]
        selected = [test for idx, test in enumerate(selected) if idx in keep_index_set]

    selection_note = (
        "Generated the exact requested number of extreme tasks from built-in generators."
        if not dropped_tests
        else "Generated more than requested from built-in generators, then deterministically dropped extras to keep the requested total without modifying project code."
    )

    return selected, {
        "difficulty": "extreme",
        "generator_tasks_per_family": tasks_per_family,
        "generated_before_trim": len(generated),
        "generated_after_trim": len(selected),
        "dropped_tests": [
            {
                "test_id": test.test_id,
                "family": test.family,
            }
            for test in dropped_tests
        ],
        "selection_note": selection_note,
    }


def build_extreme20(seed: int) -> tuple[list[mgt.TestCase], dict[str, Any]]:
    return build_extreme_suite(seed, 20)


def load_self_answers(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    parsed = json.loads(path.read_text())
    answers: dict[str, dict[str, Any]] = {}

    if isinstance(parsed, dict):
        for test_id, value in parsed.items():
            if isinstance(value, dict):
                output_text = value.get("output_text")
                elapsed_ms = value.get("elapsed_ms")
            else:
                output_text = value
                elapsed_ms = None
            answers[str(test_id)] = {
                "output_text": None if output_text is None else str(output_text),
                "elapsed_ms": elapsed_ms if isinstance(elapsed_ms, int) else None,
            }
        return answers

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            test_id = item.get("test_id")
            if test_id is None:
                continue
            elapsed_ms = item.get("elapsed_ms")
            answers[str(test_id)] = {
                "output_text": None
                if item.get("output_text") is None
                else str(item.get("output_text")),
                "elapsed_ms": elapsed_ms if isinstance(elapsed_ms, int) else None,
            }
        return answers

    raise ValueError(f"Unsupported self answers format in {path}")


def summarize_rows(rows: list[dict[str, Any]], provider_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in provider_names:
        provider_rows = [row for row in rows if row["provider"] == name]
        completed = [row for row in provider_rows if row.get("status") == "completed"]
        scored = [row for row in provider_rows if row.get("expected") is not None]
        scored_completed = [row for row in completed if row.get("expected") is not None]
        issue_counts: dict[str, int] = {}
        for row in provider_rows:
            issue = row.get("response_issue")
            if isinstance(issue, str) and issue:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        elapsed_values = [
            row["elapsed_ms"]
            for row in completed
            if isinstance(row.get("elapsed_ms"), int)
        ]
        reasoning_values = [
            row["reasoning_tokens"]
            for row in completed
            if isinstance(row.get("reasoning_tokens"), int)
        ]

        families = sorted({row["family"] for row in provider_rows})
        family_breakdown: dict[str, Any] = {}
        for family in families:
            family_rows = [row for row in provider_rows if row["family"] == family]
            family_issue_counts: dict[str, int] = {}
            for row in family_rows:
                issue = row.get("response_issue")
                if isinstance(issue, str) and issue:
                    family_issue_counts[issue] = family_issue_counts.get(issue, 0) + 1
            family_breakdown[family] = {
                "completed": sum(1 for row in family_rows if row.get("status") == "completed"),
                "correct": sum(1 for row in family_rows if row.get("correct")),
                "scored_total": sum(1 for row in family_rows if row.get("expected") is not None),
                "total": len(family_rows),
                "response_issues": family_issue_counts,
            }

        summary[name] = {
            "total": len(provider_rows),
            "completed": len(completed),
            "completion_rate_pct": round(100 * len(completed) / len(provider_rows), 1)
            if provider_rows
            else None,
            "scored_total": len(scored),
            "correct": sum(1 for row in scored if row.get("correct")),
            "accuracy_scored_all_pct": round(
                100 * sum(1 for row in scored if row.get("correct")) / len(scored), 1
            )
            if scored
            else None,
            "accuracy_scored_completed_pct": round(
                100
                * sum(1 for row in scored_completed if row.get("correct"))
                / len(scored_completed),
                1,
            )
            if scored_completed
            else None,
            "avg_elapsed_ms": round(sum(elapsed_values) / len(elapsed_values), 1)
            if elapsed_values
            else None,
            "median_reasoning_tokens": int(mgt.statistics.median(reasoning_values))
            if reasoning_values
            else None,
            "response_issues": issue_counts,
            "family_breakdown": family_breakdown,
        }
    return summary


def rank_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    ranked = []
    for provider, stats in summary.items():
        ranked.append(
            {
                "provider": provider,
                "completed": stats["completed"],
                "completion_rate_pct": stats["completion_rate_pct"],
                "correct": stats["correct"],
                "accuracy_scored_all_pct": stats["accuracy_scored_all_pct"],
                "avg_elapsed_ms": stats["avg_elapsed_ms"],
            }
        )

    ranked.sort(
        key=lambda item: (
            -1 if item["accuracy_scored_all_pct"] is None else -item["accuracy_scored_all_pct"],
            -1 if item["completion_rate_pct"] is None else -item["completion_rate_pct"],
            float("inf") if item["avg_elapsed_ms"] is None else item["avg_elapsed_ms"],
            item["provider"],
        )
    )
    for index, item in enumerate(ranked, start=1):
        item["rank"] = index
    return ranked


def build_report(
    *,
    seed: int,
    selection_meta: dict[str, Any],
    providers: list[mgt.ProviderConfig],
    provider_names: list[str],
    tests: list[mgt.TestCase],
    rows: list[dict[str, Any]],
    self_provider_name: str | None,
    self_answers_file: pathlib.Path | None,
    progress: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize_rows(rows, provider_names)
    leaderboard = rank_summary(summary)
    report_rows = mgt.build_report_rows(tests, rows, include_prompt_text=False)

    return {
        "meta": {
            "seed": seed,
            "generated_tests": len(tests),
            "selection": selection_meta,
            "providers": [
                asdict(provider) | {"api_key": "***redacted***"} for provider in providers
            ],
            "self_provider_name": self_provider_name,
            "self_answers_file": str(self_answers_file) if self_answers_file else None,
            "progress": progress,
        },
        "summary": summary,
        "leaderboard": leaderboard,
        "rows": report_rows,
    }


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    tests, selection_meta = build_extreme_suite(seed, args.total_tests)

    prompts_output = args.prompts_output
    if prompts_output is None:
        prompts_output = args.output.with_name(args.output.stem + "_prompts.json")
    checkpoint_output = args.output.with_name(args.output.stem + "_checkpoint.json")

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

    providers = [
        mgt.ProviderConfig(
            name=f"OpenRouter {args.openrouter_primary_model} (xhigh)",
            url=args.openrouter_url,
            api_key=args.openrouter_key,
            model=args.openrouter_primary_model,
            reasoning_mode="nested",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=(
                args.openrouter_primary_max_output_tokens
                if args.openrouter_primary_max_output_tokens is not None
                else args.max_output_tokens
            ),
        ),
        mgt.ProviderConfig(
            name=f"OpenRouter {args.openrouter_secondary_model}(xhigh)",
            url=args.openrouter_url,
            api_key=args.openrouter_key,
            model=args.openrouter_secondary_model,
            reasoning_mode="nested",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        ),
        mgt.ProviderConfig(
            name=f"Longent {args.longent_primary_model}",
            url=args.longent_url,
            api_key=args.longent_key,
            model=args.longent_primary_model,
            reasoning_mode="none",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            needs_user_agent=True,
        ),
        mgt.ProviderConfig(
            name=f"Longent {args.longent_secondary_model}",
            url=args.longent_url,
            api_key=args.longent_key,
            model=args.longent_secondary_model,
            reasoning_mode="none",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            needs_user_agent=True,
        ),
    ]

    rows: list[dict[str, Any]] = []
    total_requests = len(tests) * len(providers)
    request_index = 0
    for test_index, test in enumerate(tests, start=1):
        print(
            f"[test {test.test_id}] dispatching {len(providers)} providers",
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
                request_index += 1
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
                    f"[{request_index}/{total_requests}] {provider.name} :: {test.test_id}",
                    file=sys.stderr,
                    flush=True,
                )

        for provider in providers:
            rows.append(completed_for_test[provider.name])

        partial_provider_names = [provider.name for provider in providers]
        checkpoint_report = build_report(
            seed=seed,
            selection_meta=selection_meta,
            providers=providers,
            provider_names=partial_provider_names,
            tests=tests,
            rows=rows,
            self_provider_name=None,
            self_answers_file=None,
            progress={
                "completed_tests": test_index,
                "total_tests": len(tests),
                "completed_requests": request_index,
                "total_requests": total_requests,
                "status": "running",
            },
        )
        checkpoint_output.write_text(json.dumps(checkpoint_report, ensure_ascii=False, indent=2))

    if args.self_answers_file is not None:
        self_answers = load_self_answers(args.self_answers_file)
        for test in tests:
            self_row = self_answers.get(test.test_id, {})
            output_text = self_row.get("output_text")
            status = "completed" if output_text is not None else "missing"
            rows.append(
                {
                    "test": test.test_id,
                    "family": test.family,
                    "provider": args.self_provider_name,
                    "expected": test.expected,
                    "status": status,
                    "elapsed_ms": self_row.get("elapsed_ms"),
                    "returncode": 0 if output_text is not None else None,
                    "returned_model": None,
                    "returned_reasoning_effort": None,
                    "returned_service_tier": None,
                    "reasoning_tokens": None,
                    "output_text": output_text,
                    "correct": output_text == test.expected if output_text is not None else False,
                }
            )

    provider_names = [provider.name for provider in providers]
    if args.self_answers_file is not None:
        provider_names.append(args.self_provider_name)

    report = build_report(
        seed=seed,
        selection_meta=selection_meta,
        providers=providers,
        provider_names=provider_names,
        tests=tests,
        rows=rows,
        self_provider_name=args.self_provider_name if args.self_answers_file is not None else None,
        self_answers_file=args.self_answers_file,
        progress={
            "completed_tests": len(tests),
            "total_tests": len(tests),
            "completed_requests": total_requests,
            "total_requests": total_requests,
            "status": "completed",
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    checkpoint_output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    print(
        json.dumps(
            {
                "output": str(args.output),
                "checkpoint_output": str(checkpoint_output),
                "prompts_output": str(prompts_output),
                "leaderboard": report["leaderboard"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
