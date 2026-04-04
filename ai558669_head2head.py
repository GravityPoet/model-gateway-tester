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

import fiveway_extreme20 as fx
import model_gateway_tester as mgt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fresh fair benchmark for two to six models on OpenAI-compatible endpoints."
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-tests", type=int, default=30)
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://ai.558669.xyz/v1/responses",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default="gpt-5.4",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="gpt-5.4-codex-xhigh",
    )
    parser.add_argument(
        "--name-a",
        type=str,
        default="558669xyz gpt-5.4 (xhigh)",
    )
    parser.add_argument(
        "--name-b",
        type=str,
        default="558669xyz gpt-5.4-codex-xhigh",
    )
    parser.add_argument(
        "--base-url-b",
        type=str,
        default=None,
        help="Optional override for model-b base URL.",
    )
    parser.add_argument(
        "--api-key-b",
        type=str,
        default=None,
        help="Optional override for model-b API key.",
    )
    parser.add_argument(
        "--model-c",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name-c",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base-url-c",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key-c",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-d",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name-d",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base-url-d",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key-d",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-e",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name-e",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base-url-e",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key-e",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model-f",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--name-f",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--base-url-f",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--api-key-f",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="xhigh",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd()
        / f"ai558669_head2head_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument(
        "--prompts-output",
        type=pathlib.Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    tests, selection_meta = fx.build_extreme_suite(seed, args.total_tests)

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
            name=args.name_a,
            url=args.base_url,
            api_key=args.api_key,
            model=args.model_a,
            reasoning_mode="nested",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        ),
        mgt.ProviderConfig(
            name=args.name_b,
            url=args.base_url_b or args.base_url,
            api_key=args.api_key_b or args.api_key,
            model=args.model_b,
            reasoning_mode="nested",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        ),
    ]
    if args.model_c:
        providers.append(
            mgt.ProviderConfig(
                name=args.name_c or args.model_c,
                url=args.base_url_c or args.base_url,
                api_key=args.api_key_c or args.api_key,
                model=args.model_c,
                reasoning_mode="nested",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
    if args.model_d:
        providers.append(
            mgt.ProviderConfig(
                name=args.name_d or args.model_d,
                url=args.base_url_d or args.base_url,
                api_key=args.api_key_d or args.api_key,
                model=args.model_d,
                reasoning_mode="nested",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
    if args.model_e:
        providers.append(
            mgt.ProviderConfig(
                name=args.name_e or args.model_e,
                url=args.base_url_e or args.base_url,
                api_key=args.api_key_e or args.api_key,
                model=args.model_e,
                reasoning_mode="nested",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )
    if args.model_f:
        providers.append(
            mgt.ProviderConfig(
                name=args.name_f or args.model_f,
                url=args.base_url_f or args.base_url,
                api_key=args.api_key_f or args.api_key,
                model=args.model_f,
                reasoning_mode="nested",
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )

    rows: list[dict[str, object]] = []
    total_requests = len(tests) * len(providers)
    completed_requests = 0

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
            completed_for_test: dict[str, dict[str, object]] = {}
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
                    f"[{completed_requests}/{total_requests}] {provider.name} :: {test.test_id}",
                    file=sys.stderr,
                    flush=True,
                )
        for provider in providers:
            rows.append(completed_for_test[provider.name])

        checkpoint_report = fx.build_report(
            seed=seed,
            selection_meta=selection_meta,
            providers=providers,
            provider_names=[provider.name for provider in providers],
            tests=tests,
            rows=rows,
            self_provider_name=None,
            self_answers_file=None,
            progress={
                "completed_tests": test_index,
                "total_tests": len(tests),
                "completed_requests": completed_requests,
                "total_requests": total_requests,
                "status": "running",
            },
        )
        checkpoint_report["meta"]["note"] = (
            "Fresh fair benchmark on OpenAI-compatible endpoints with every model using "
            "the same reasoning.effort and the same max_output_tokens."
        )
        checkpoint_output.write_text(json.dumps(checkpoint_report, ensure_ascii=False, indent=2))

    provider_names = [provider.name for provider in providers]
    report = fx.build_report(
        seed=seed,
        selection_meta=selection_meta,
        providers=providers,
        provider_names=provider_names,
        tests=tests,
        rows=rows,
        self_provider_name=None,
        self_answers_file=None,
        progress={
            "completed_tests": len(tests),
            "total_tests": len(tests),
            "completed_requests": total_requests,
            "total_requests": total_requests,
            "status": "completed",
        },
    )
    report["meta"]["note"] = (
        "Fresh fair benchmark on OpenAI-compatible endpoints with every model using "
        "the same reasoning.effort and the same max_output_tokens."
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
