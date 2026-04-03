#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import secrets
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import comb, log2
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore


@dataclass
class TestCase:
    test_id: str
    family: str
    prompt: str
    expected: str | None
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderConfig:
    name: str
    url: str
    api_key: str
    model: str
    reasoning_mode: str
    reasoning_effort: str
    max_output_tokens: int
    needs_user_agent: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Short-cycle shadow audit for third-party LLM gateways."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. If omitted, a fresh random seed is generated for every run.",
    )
    parser.add_argument(
        "--tasks-per-family",
        type=int,
        default=10,
        help="Number of dynamic private variants to generate per family. Use 0 for shadow-only mode.",
    )
    parser.add_argument(
        "--difficulty",
        choices=["standard", "mixed", "hard", "very-hard", "extreme"],
        default="extreme",
        help="Dynamic task difficulty profile. standard=old easier families, hard=new harder families, very-hard=top-tier separator tasks, extreme=maximum built-in stress tier, mixed=standard+hard.",
    )
    parser.add_argument(
        "--prompt-file",
        type=pathlib.Path,
        default=None,
        help="Optional shadow prompt file (.jsonl, .json, .txt, .md).",
    )
    parser.add_argument(
        "--shadow-sample-size",
        type=int,
        default=0,
        help="If > 0, randomly sample this many prompts from --prompt-file.",
    )
    parser.add_argument(
        "--include-prompt-text",
        action="store_true",
        help="Include raw prompt text in the report. Off by default to avoid leaking real prompts.",
    )
    parser.add_argument(
        "--openrouter-key",
        type=str,
        default="",
        help="OpenRouter API key. If omitted, reads OPENROUTER_API_KEY from env is not supported; pass it explicitly.",
    )
    parser.add_argument(
        "--openrouter-model",
        type=str,
        default="openai/gpt-5.4",
        help="OpenRouter model slug.",
    )
    parser.add_argument(
        "--openrouter-url",
        type=str,
        default="https://openrouter.ai/api/v1/responses",
        help="OpenRouter Responses endpoint.",
    )
    parser.add_argument(
        "--longent-model",
        type=str,
        default="gpt-5.4-fast(xhigh)",
        help="Longent model slug.",
    )
    parser.add_argument(
        "--fireworks-key",
        type=str,
        default="",
        help="Optional Fireworks API key.",
    )
    parser.add_argument(
        "--fireworks-model",
        type=str,
        default="accounts/fireworks/routers/kimi-k2p5-turbo",
        help="Optional Fireworks model slug.",
    )
    parser.add_argument(
        "--fireworks-url",
        type=str,
        default="https://api.fireworks.ai/inference/v1/responses",
        help="Optional Fireworks Responses endpoint.",
    )
    parser.add_argument(
        "--fireworks-reasoning-effort",
        type=str,
        default="high",
        help="Fireworks highest public reasoning_effort currently supported on this route.",
    )
    parser.add_argument(
        "--longent-config",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".codex" / "config.toml",
        help="Path to config.toml used to read the Longent bearer token.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="max_output_tokens for Responses requests.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="xhigh",
        help="Reasoning effort for providers that accept explicit reasoning config.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path.cwd()
        / f"model_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        help="Path to the JSON report.",
    )
    return parser.parse_args()


def load_longent_key(config_path: pathlib.Path) -> str:
    if tomllib is None:
        raise ModuleNotFoundError(
            "tomllib/tomli is required to read the Longent config file"
        )
    data = tomllib.loads(config_path.read_text())
    key = (
        data.get("model_providers", {})
        .get("longent", {})
        .get("experimental_bearer_token", "")
    )
    if not isinstance(key, str) or not key:
        raise ValueError(
            f"Could not find model_providers.longent.experimental_bearer_token in {config_path}"
        )
    return key


def difficulty_family_count(difficulty: str) -> int:
    if difficulty == "standard":
        return 6
    if difficulty == "hard":
        return 3
    if difficulty == "very-hard":
        return 3
    if difficulty == "extreme":
        return 10
    if difficulty == "mixed":
        return 9
    raise ValueError(f"Unknown difficulty: {difficulty}")


def build_dynamic_test_suite(
    seed: int, tasks_per_family: int, difficulty: str
) -> list[TestCase]:
    rng = random.Random(seed)
    tests: list[TestCase] = []

    def mk_py_trace(idx: int) -> TestCase:
        n = rng.randint(6, 11)
        add_mul = rng.randint(2, 4)
        even_sub = rng.randint(1, 3)
        odd_add = rng.randint(1, 3)
        base = rng.randint(-2, 3)
        total = base
        for i in range(1, n + 1):
            if i % 3 == 0:
                total += i * add_mul
            elif i % 2 == 0:
                total -= i * even_sub
            else:
                total += i * odd_add
        prompt = (
            f"Without showing work, what integer does this Python function return for n={n}?\n\n"
            f"def f(n):\n"
            f"    s = {base}\n"
            f"    for i in range(1, n + 1):\n"
            f"        if i % 3 == 0:\n"
            f"            s += i * {add_mul}\n"
            f"        elif i % 2 == 0:\n"
            f"            s -= i * {even_sub}\n"
            f"        else:\n"
            f"            s += i * {odd_add}\n"
            f"    return s\n\n"
            f"Reply with only the integer."
        )
        return TestCase(
            test_id=f"py_trace_{idx}",
            family="py_trace",
            prompt=prompt,
            expected=str(total),
            source="dynamic",
        )

    def mk_path(idx: int) -> TestCase:
        moves = [("U", (0, 1)), ("D", (0, -1)), ("L", (-1, 0)), ("R", (1, 0))]
        x = rng.randint(-2, 2)
        y = rng.randint(-2, 2)
        sx, sy = x, y
        steps: list[str] = []
        for _ in range(6):
            move, (dx, dy) = rng.choice(moves)
            k = rng.randint(1, 5)
            steps.append(f"{move}{k}")
            x += dx * k
            y += dy * k
        prompt = (
            f"Start at ({sx},{sy}). Apply moves {', '.join(steps)}. "
            f"Reply only x,y with no spaces."
        )
        return TestCase(
            test_id=f"path_{idx}",
            family="path",
            prompt=prompt,
            expected=f"{x},{y}",
            source="dynamic",
        )

    def mk_combinatorics(idx: int) -> TestCase:
        length = rng.randint(4, 6)
        target = rng.randint(1, length - 1)
        total = comb(length, target) * (2 ** (length - target))
        prompt = (
            f"Reply with only the integer. How many {length}-letter strings "
            f"over {{A,B,C}} contain exactly {target} As?"
        )
        return TestCase(
            test_id=f"comb_{idx}",
            family="comb",
            prompt=prompt,
            expected=str(total),
            source="dynamic",
        )

    def mk_json_extract(idx: int) -> TestCase:
        keys = rng.sample(list("pqrstuvw"), 3)
        vals = rng.sample(range(7, 30), 3)
        obj = {keys[0]: vals[0], keys[1]: vals[1], keys[2]: vals[2]}
        expected = json.dumps(obj, separators=(",", ":"))
        prompt = (
            f"Return only minified JSON with keys {keys[0]},{keys[1]},{keys[2]}. "
            f"Source text: {keys[0]}={vals[0]}; {keys[1]}={vals[1]}; {keys[2]}={vals[2]}. "
            f"Output exactly {expected}"
        )
        return TestCase(
            test_id=f"json_{idx}",
            family="json",
            prompt=prompt,
            expected=expected,
            source="dynamic",
        )

    def mk_transform(idx: int) -> TestCase:
        arr = rng.sample(range(1, 9), 4)
        out = [n + 3 if n % 2 else n - 1 for n in arr]
        out.sort()
        expected = ",".join(str(x) for x in out)
        prompt = (
            f"Take the list {arr}. Add 3 to each odd number, subtract 1 from each even number, "
            f"then sort ascending. Reply only as comma-separated integers."
        )
        return TestCase(
            test_id=f"transform_{idx}",
            family="transform",
            prompt=prompt,
            expected=expected,
            source="dynamic",
        )

    def mk_digitsum(idx: int) -> TestCase:
        lo = rng.randint(10, 30)
        hi = lo + rng.randint(15, 25)
        target = rng.randint(5, 11)
        cnt = sum(1 for n in range(lo, hi + 1) if sum(map(int, str(n))) == target)
        prompt = (
            f"Reply with only the integer. How many integers from {lo} to {hi} inclusive "
            f"have digit sum equal to {target}?"
        )
        return TestCase(
            test_id=f"digitsum_{idx}",
            family="digitsum",
            prompt=prompt,
            expected=str(cnt),
            source="dynamic",
        )

    def mk_register_machine(idx: int) -> TestCase:
        nums = rng.sample(range(2, 10), rng.randint(5, 7))
        base = rng.randint(-3, 4)
        mul = rng.randint(2, 3)
        sub = rng.randint(2, 4)
        flag = rng.randint(0, 2)
        acc = base
        for x in nums:
            if (x + flag) % 2 == 0:
                acc = acc * mul + x
                flag = (flag + 1) % 3
            elif x % 3 == 0:
                acc = acc - x * sub
                flag = (flag + 2) % 3
            else:
                acc = acc + x + flag
        prompt = (
            "Without showing work, what integer does this Python function return?\n\n"
            "def run():\n"
            f"    acc = {base}\n"
            f"    flag = {flag}\n"
            f"    nums = {nums}\n"
            "    for x in nums:\n"
            "        if (x + flag) % 2 == 0:\n"
            f"            acc = acc * {mul} + x\n"
            "            flag = (flag + 1) % 3\n"
            "        elif x % 3 == 0:\n"
            f"            acc = acc - x * {sub}\n"
            "            flag = (flag + 2) % 3\n"
            "        else:\n"
            "            acc = acc + x + flag\n"
            "    return acc\n\n"
            "Reply with only the integer."
        )
        return TestCase(
            test_id=f"register_machine_{idx}",
            family="register_machine",
            prompt=prompt,
            expected=str(acc),
            source="dynamic",
        )

    def mk_table_query(idx: int) -> TestCase:
        teams = ["red", "blue", "green"]
        rows = []
        for i in range(rng.randint(5, 6)):
            rows.append(
                {
                    "name": chr(ord("A") + i),
                    "team": rng.choice(teams),
                    "score": rng.randint(2, 9),
                    "weight": rng.randint(1, 4),
                }
            )
        target_team = rng.choice(teams)
        threshold = rng.randint(4, 7)
        selected = [
            row for row in rows if row["team"] != target_team and row["score"] >= threshold
        ]
        selected.sort(key=lambda row: (row["weight"], row["name"]))
        picked = selected[:3]
        total = sum(row["score"] * row["weight"] for row in picked)
        prompt_rows = "; ".join(
            f"{row['name']} team={row['team']} score={row['score']} weight={row['weight']}"
            for row in rows
        )
        prompt = (
            "Reply with only the integer. Rows: "
            f"{prompt_rows}. "
            f"Keep rows where team != {target_team} and score >= {threshold}. "
            "Sort remaining rows by weight ascending, then name ascending. "
            "Take the first 3 rows after sorting. "
            "Return sum(score * weight) over those rows."
        )
        return TestCase(
            test_id=f"table_query_{idx}",
            family="table_query",
            prompt=prompt,
            expected=str(total),
            source="dynamic",
        )

    def mk_fsm(idx: int) -> TestCase:
        states = ["A", "B", "C", "D"]
        tokens = rng.choices(["x", "y", "z"], k=rng.randint(6, 8))
        state = rng.choice(states)
        counter = rng.randint(0, 3)
        start_state = state
        start_counter = counter
        for token in tokens:
            if state == "A":
                if token == "x":
                    state, counter = "B", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 1
                else:
                    state, counter = "D", counter + 1
            elif state == "B":
                if token == "x":
                    state, counter = "B", counter + 1
                elif token == "y":
                    state, counter = "D", counter + 2
                else:
                    state, counter = "A", counter - 2
            elif state == "C":
                if token == "x":
                    state, counter = "D", counter + 3
                elif token == "y":
                    state, counter = "A", counter + 1
                else:
                    state, counter = "C", counter - 1
            else:
                if token == "x":
                    state, counter = "A", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 2
                else:
                    state, counter = "B", counter + 1
        prompt = (
            f"Start in state {start_state} with counter={start_counter}. "
            f"Process tokens in order: {','.join(tokens)}. "
            "Transition rules: "
            "A:x->(B,+2), A:y->(C,-1), A:z->(D,+1); "
            "B:x->(B,+1), B:y->(D,+2), B:z->(A,-2); "
            "C:x->(D,+3), C:y->(A,+1), C:z->(C,-1); "
            "D:x->(A,+2), D:y->(C,-2), D:z->(B,+1). "
            "Reply only as STATE,COUNTER."
        )
        return TestCase(
            test_id=f"fsm_{idx}",
            family="fsm",
            prompt=prompt,
            expected=f"{state},{counter}",
            source="dynamic",
        )

    def mk_register_machine_long(idx: int) -> TestCase:
        nums = rng.sample(range(2, 18), rng.randint(10, 14))
        base = rng.randint(-8, 8)
        mul = rng.randint(2, 5)
        sub = rng.randint(2, 6)
        bonus = rng.randint(1, 4)
        flag = rng.randint(0, 3)
        acc = base
        start_flag = flag
        for x in nums:
            if (x + flag) % 4 == 0:
                acc = acc * mul + x - flag
                flag = (flag + 1) % 4
            elif x % 5 == 0:
                acc = acc - x * sub + bonus
                flag = (flag + 2) % 4
            elif x % 2 == 0:
                acc = acc + x * bonus - flag
            else:
                acc = acc + x + flag + bonus
        prompt = (
            "Without showing work, what integer does this Python function return?\n\n"
            "def run():\n"
            f"    acc = {base}\n"
            f"    flag = {start_flag}\n"
            f"    nums = {nums}\n"
            "    for x in nums:\n"
            "        if (x + flag) % 4 == 0:\n"
            f"            acc = acc * {mul} + x - flag\n"
            "            flag = (flag + 1) % 4\n"
            "        elif x % 5 == 0:\n"
            f"            acc = acc - x * {sub} + {bonus}\n"
            "            flag = (flag + 2) % 4\n"
            "        elif x % 2 == 0:\n"
            f"            acc = acc + x * {bonus} - flag\n"
            "        else:\n"
            f"            acc = acc + x + flag + {bonus}\n"
            "    return acc\n\n"
            "Reply with only the integer."
        )
        return TestCase(
            test_id=f"register_machine_long_{idx}",
            family="register_machine_long",
            prompt=prompt,
            expected=str(acc),
            source="dynamic",
        )

    def mk_table_query_hard(idx: int) -> TestCase:
        teams = ["red", "blue", "green", "gold"]
        regions = ["east", "west", "north", "south"]
        rows = []
        for i in range(rng.randint(8, 10)):
            rows.append(
                {
                    "name": chr(ord("A") + i),
                    "team": rng.choice(teams),
                    "region": rng.choice(regions),
                    "score": rng.randint(2, 15),
                    "weight": rng.randint(1, 5),
                    "bonus": rng.randint(0, 4),
                }
            )
        target_team = rng.choice(teams)
        target_region = rng.choice(regions)
        threshold = rng.randint(6, 10)
        selected = [
            row
            for row in rows
            if row["team"] != target_team
            and row["region"] != target_region
            and row["score"] >= threshold
        ]
        selected.sort(key=lambda row: (row["weight"], -row["score"], row["name"]))
        picked = selected[:4]
        total = sum((row["score"] + row["bonus"]) * row["weight"] for row in picked)
        prompt_rows = "; ".join(
            f"{row['name']} team={row['team']} region={row['region']} score={row['score']} weight={row['weight']} bonus={row['bonus']}"
            for row in rows
        )
        prompt = (
            "Reply with only the integer. Rows: "
            f"{prompt_rows}. "
            f"Keep rows where team != {target_team}, region != {target_region}, and score >= {threshold}. "
            "Sort remaining rows by weight ascending, then score descending, then name ascending. "
            "Take the first 4 rows after sorting. "
            "Return sum((score + bonus) * weight) over those rows."
        )
        return TestCase(
            test_id=f"table_query_hard_{idx}",
            family="table_query_hard",
            prompt=prompt,
            expected=str(total),
            source="dynamic",
        )

    def mk_fsm_long(idx: int) -> TestCase:
        states = ["A", "B", "C", "D", "E"]
        tokens = rng.choices(["x", "y", "z"], k=rng.randint(12, 16))
        state = rng.choice(states)
        counter = rng.randint(-2, 4)
        start_state = state
        start_counter = counter
        for token in tokens:
            if state == "A":
                if token == "x":
                    state, counter = "B", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 1
                else:
                    state, counter = "D", counter + 1
            elif state == "B":
                if token == "x":
                    state, counter = "E", counter + 1
                elif token == "y":
                    state, counter = "D", counter + 2
                else:
                    state, counter = "A", counter - 2
            elif state == "C":
                if token == "x":
                    state, counter = "D", counter + 3
                elif token == "y":
                    state, counter = "A", counter + 1
                else:
                    state, counter = "C", counter - 1
            elif state == "D":
                if token == "x":
                    state, counter = "A", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 2
                else:
                    state, counter = "B", counter + 1
            else:
                if token == "x":
                    state, counter = "C", counter + 2
                elif token == "y":
                    state, counter = "E", counter - 1
                else:
                    state, counter = "B", counter + 3
        prompt = (
            f"Start in state {start_state} with counter={start_counter}. "
            f"Process tokens in order: {','.join(tokens)}. "
            "Transition rules: "
            "A:x->(B,+2), A:y->(C,-1), A:z->(D,+1); "
            "B:x->(E,+1), B:y->(D,+2), B:z->(A,-2); "
            "C:x->(D,+3), C:y->(A,+1), C:z->(C,-1); "
            "D:x->(A,+2), D:y->(C,-2), D:z->(B,+1); "
            "E:x->(C,+2), E:y->(E,-1), E:z->(B,+3). "
            "Reply only as STATE,COUNTER."
        )
        return TestCase(
            test_id=f"fsm_long_{idx}",
            family="fsm_long",
            prompt=prompt,
            expected=f"{state},{counter}",
            source="dynamic",
        )

    def mk_register_machine_extreme(idx: int) -> TestCase:
        nums = rng.sample(range(2, 24), rng.randint(14, 18))
        base = rng.randint(-10, 10)
        mul = rng.randint(2, 6)
        sub = rng.randint(2, 7)
        bonus = rng.randint(1, 5)
        flag = rng.randint(0, 4)
        acc = base
        start_flag = flag
        for x in nums:
            if (x + flag) % 5 == 0:
                acc = acc * mul + x - flag
                flag = (flag + 1) % 5
            elif x % 7 == 0:
                acc = acc - x * sub + bonus
                flag = (flag + 3) % 5
            elif x % 3 == 0:
                acc = acc + x * bonus - (flag * 2)
            elif x % 2 == 0:
                acc = acc - x + bonus + flag
            else:
                acc = acc + x + flag + bonus
        prompt = (
            "Without showing work, what integer does this Python function return?\n\n"
            "def run():\n"
            f"    acc = {base}\n"
            f"    flag = {start_flag}\n"
            f"    nums = {nums}\n"
            "    for x in nums:\n"
            "        if (x + flag) % 5 == 0:\n"
            f"            acc = acc * {mul} + x - flag\n"
            "            flag = (flag + 1) % 5\n"
            "        elif x % 7 == 0:\n"
            f"            acc = acc - x * {sub} + {bonus}\n"
            "            flag = (flag + 3) % 5\n"
            "        elif x % 3 == 0:\n"
            f"            acc = acc + x * {bonus} - (flag * 2)\n"
            "        elif x % 2 == 0:\n"
            f"            acc = acc - x + {bonus} + flag\n"
            "        else:\n"
            f"            acc = acc + x + flag + {bonus}\n"
            "    return acc\n\n"
            "Reply with only the integer."
        )
        return TestCase(
            test_id=f"register_machine_extreme_{idx}",
            family="register_machine_extreme",
            prompt=prompt,
            expected=str(acc),
            source="dynamic",
        )

    def mk_table_query_extreme(idx: int) -> TestCase:
        teams = ["red", "blue", "green", "gold", "black"]
        regions = ["east", "west", "north", "south", "central"]
        rows = []
        for i in range(rng.randint(10, 13)):
            rows.append(
                {
                    "name": chr(ord("A") + i),
                    "team": rng.choice(teams),
                    "region": rng.choice(regions),
                    "score": rng.randint(2, 18),
                    "weight": rng.randint(1, 6),
                    "bonus": rng.randint(0, 5),
                    "penalty": rng.randint(0, 3),
                }
            )
        target_team = rng.choice(teams)
        target_region = rng.choice(regions)
        threshold = rng.randint(7, 12)
        selected = [
            row
            for row in rows
            if row["team"] != target_team
            and row["region"] != target_region
            and row["score"] >= threshold
            and row["bonus"] >= row["penalty"]
        ]
        selected.sort(
            key=lambda row: (row["weight"], -(row["score"] + row["bonus"]), row["name"])
        )
        picked = selected[:5]
        total = sum(
            (row["score"] + row["bonus"] - row["penalty"]) * row["weight"]
            for row in picked
        )
        prompt_rows = "; ".join(
            f"{row['name']} team={row['team']} region={row['region']} score={row['score']} weight={row['weight']} bonus={row['bonus']} penalty={row['penalty']}"
            for row in rows
        )
        prompt = (
            "Reply with only the integer. Rows: "
            f"{prompt_rows}. "
            f"Keep rows where team != {target_team}, region != {target_region}, score >= {threshold}, and bonus >= penalty. "
            "Sort remaining rows by weight ascending, then (score + bonus) descending, then name ascending. "
            "Take the first 5 rows after sorting. "
            "Return sum((score + bonus - penalty) * weight) over those rows."
        )
        return TestCase(
            test_id=f"table_query_extreme_{idx}",
            family="table_query_extreme",
            prompt=prompt,
            expected=str(total),
            source="dynamic",
        )

    def mk_fsm_extreme(idx: int) -> TestCase:
        states = ["A", "B", "C", "D", "E", "F"]
        tokens = rng.choices(["x", "y", "z"], k=rng.randint(16, 20))
        state = rng.choice(states)
        counter = rng.randint(-3, 5)
        start_state = state
        start_counter = counter
        for token in tokens:
            if state == "A":
                if token == "x":
                    state, counter = "B", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 1
                else:
                    state, counter = "D", counter + 1
            elif state == "B":
                if token == "x":
                    state, counter = "E", counter + 1
                elif token == "y":
                    state, counter = "D", counter + 2
                else:
                    state, counter = "A", counter - 2
            elif state == "C":
                if token == "x":
                    state, counter = "D", counter + 3
                elif token == "y":
                    state, counter = "A", counter + 1
                else:
                    state, counter = "C", counter - 1
            elif state == "D":
                if token == "x":
                    state, counter = "A", counter + 2
                elif token == "y":
                    state, counter = "C", counter - 2
                else:
                    state, counter = "B", counter + 1
            elif state == "E":
                if token == "x":
                    state, counter = "C", counter + 2
                elif token == "y":
                    state, counter = "E", counter - 1
                else:
                    state, counter = "F", counter + 3
            else:
                if token == "x":
                    state, counter = "B", counter - 1
                elif token == "y":
                    state, counter = "D", counter + 4
                else:
                    state, counter = "A", counter - 3
        prompt = (
            f"Start in state {start_state} with counter={start_counter}. "
            f"Process tokens in order: {','.join(tokens)}. "
            "Transition rules: "
            "A:x->(B,+2), A:y->(C,-1), A:z->(D,+1); "
            "B:x->(E,+1), B:y->(D,+2), B:z->(A,-2); "
            "C:x->(D,+3), C:y->(A,+1), C:z->(C,-1); "
            "D:x->(A,+2), D:y->(C,-2), D:z->(B,+1); "
            "E:x->(C,+2), E:y->(E,-1), E:z->(F,+3); "
            "F:x->(B,-1), F:y->(D,+4), F:z->(A,-3). "
            "Reply only as STATE,COUNTER."
        )
        return TestCase(
            test_id=f"fsm_extreme_{idx}",
            family="fsm_extreme",
            prompt=prompt,
            expected=f"{state},{counter}",
            source="dynamic",
        )

    def mk_stack_machine_extreme(idx: int) -> TestCase:
        stack = rng.sample(range(2, 12), 3)
        start_stack = list(stack)
        ops: list[str] = []

        for _ in range(rng.randint(15, 19)):
            valid_ops = ["PUSH", "DUP", "TWIST"]
            if len(stack) >= 2:
                valid_ops.extend(["SWAP", "ADD", "SUB", "MULADD"])
            if len(stack) >= 3:
                valid_ops.append("ROT")
            if len(stack) >= 7:
                valid_ops = [op for op in valid_ops if op not in {"PUSH", "DUP"}]

            op = rng.choice(valid_ops)
            if op == "PUSH":
                value = rng.randint(2, 13)
                stack.append(value)
                ops.append(f"PUSH{value}")
            elif op == "DUP":
                stack.append(stack[-1])
                ops.append("DUP")
            elif op == "TWIST":
                k = rng.randint(1, 4)
                x = stack.pop()
                stack.append(x // 2 + k if x % 2 == 0 else x * 2 - k)
                ops.append(f"TWIST{k}")
            elif op == "SWAP":
                stack[-1], stack[-2] = stack[-2], stack[-1]
                ops.append("SWAP")
            elif op == "ROT":
                a, b, c = stack[-3:]
                stack[-3:] = [b, c, a]
                ops.append("ROT")
            elif op == "ADD":
                b = stack.pop()
                a = stack.pop()
                stack.append(a + b)
                ops.append("ADD")
            elif op == "SUB":
                b = stack.pop()
                a = stack.pop()
                stack.append(a - b)
                ops.append("SUB")
            else:
                k = rng.randint(1, 5)
                b = stack.pop()
                a = stack.pop()
                stack.append(a * b + k)
                ops.append(f"MULADD{k}")

        expected = ",".join(str(x) for x in reversed(stack))
        prompt = (
            f"Start with stack bottom->top {start_stack}. Apply operations in order: {', '.join(ops)}. "
            "Rules: PUSHn pushes n; DUP duplicates top; SWAP swaps top two; ROT turns top three "
            "from [a,b,c] into [b,c,a]; ADD pops a,b and pushes a+b; SUB pops a,b and pushes a-b "
            "where b is the top and a is the next item; MULADDk pops a,b and pushes a*b+k; "
            "TWISTk replaces top x with x//2+k if x is even, otherwise x*2-k. "
            "Reply only with the final stack top->bottom as comma-separated integers."
        )
        return TestCase(
            test_id=f"stack_machine_extreme_{idx}",
            family="stack_machine_extreme",
            prompt=prompt,
            expected=expected,
            source="dynamic",
        )

    def mk_window_scan_extreme(idx: int) -> TestCase:
        arr = [rng.randint(2, 19) for _ in range(rng.randint(16, 21))]
        window = rng.randint(5, 7)
        best_start = 0
        best_score: int | None = None
        for start in range(0, len(arr) - window + 1):
            window_values = arr[start : start + window]
            score = (
                sum(x * 2 for x in window_values if x % 2 == 1)
                - sum(x for x in window_values if x % 2 == 0)
                + sum(1 for x in window_values if x % 3 == 0)
                + (max(window_values) - min(window_values))
            )
            if best_score is None or score > best_score:
                best_start = start
                best_score = score
        prompt = (
            f"Given the list {arr}, examine every contiguous window of length {window}. "
            "For a window, define score = 2*sum(odd values) - sum(even values) + "
            "count(values divisible by 3) + (max(window) - min(window)). "
            "Pick the highest-scoring window; if there is a tie, pick the smallest 1-indexed start. "
            "Reply only as START,SCORE."
        )
        return TestCase(
            test_id=f"window_scan_extreme_{idx}",
            family="window_scan_extreme",
            prompt=prompt,
            expected=f"{best_start + 1},{best_score}",
            source="dynamic",
        )

    def mk_json_contract_extreme(idx: int) -> TestCase:
        tier_weight = {"bronze": 1, "silver": 2, "gold": 3}
        statuses = ["live", "hold", "drop", "review"]
        tiers = ["bronze", "silver", "gold"]
        rows = []
        for i in range(rng.randint(6, 8)):
            rows.append(
                {
                    "id": chr(ord("A") + i),
                    "status": rng.choice(statuses),
                    "tier": rng.choice(tiers),
                    "qty": rng.randint(3, 11),
                    "bonus": rng.randint(0, 5),
                    "penalty": rng.randint(0, 3),
                }
            )

        threshold = rng.randint(6, 10)
        selected = [
            row
            for row in rows
            if row["status"] != "drop"
            and row["qty"] >= threshold
            and row["bonus"] >= row["penalty"]
        ]
        selected.sort(
            key=lambda row: (
                -tier_weight[row["tier"]],
                -(row["qty"] + row["bonus"] - row["penalty"]),
                row["id"],
            )
        )
        winner = selected[0]["id"] if selected else "NONE"
        score = sum(
            (row["qty"] + row["bonus"] - row["penalty"]) * tier_weight[row["tier"]]
            for row in selected
        )
        ids = "|".join(row["id"] for row in selected[:4])
        expected = json.dumps(
            {
                "count": len(selected),
                "ids": ids,
                "score": score,
                "winner": winner,
            },
            separators=(",", ":"),
        )
        prompt_rows = "; ".join(
            f"{row['id']} status={row['status']} tier={row['tier']} qty={row['qty']} bonus={row['bonus']} penalty={row['penalty']}"
            for row in rows
        )
        prompt = (
            "Return only minified JSON with keys count,ids,score,winner in that exact order. "
            f"Rows: {prompt_rows}. "
            f"Keep rows where status != drop, qty >= {threshold}, and bonus >= penalty. "
            "Sort remaining rows by tier rank gold>silver>bronze, then by "
            "(qty + bonus - penalty) descending, then id ascending. "
            "Set winner to the first id after sorting, or NONE if nothing remains. "
            "Set ids to the first 4 remaining ids joined by |. "
            "Set score to sum((qty + bonus - penalty) * tierWeight), where bronze=1, silver=2, gold=3."
        )
        return TestCase(
            test_id=f"json_contract_extreme_{idx}",
            family="json_contract_extreme",
            prompt=prompt,
            expected=expected,
            source="dynamic",
        )

    def mk_dependency_schedule_extreme(idx: int) -> TestCase:
        names = list("ABCDEFG")
        tasks = []
        for i, name in enumerate(names):
            dep_count = 0 if i == 0 else rng.randint(0, min(2, i))
            deps = sorted(rng.sample(names[:i], dep_count))
            tasks.append(
                {
                    "name": name,
                    "duration": rng.randint(1, 5),
                    "deps": deps,
                }
            )

        task_map = {task["name"]: task for task in tasks}
        completed: list[str] = []
        started: set[str] = set()
        in_progress: list[tuple[int, str]] = []
        time_cursor = 0

        while len(completed) < len(tasks):
            available = sorted(
                task["name"]
                for task in tasks
                if task["name"] not in started
                and all(dep in completed for dep in task["deps"])
            )
            while len(in_progress) < 2 and available:
                task_name = available.pop(0)
                started.add(task_name)
                in_progress.append((time_cursor + task_map[task_name]["duration"], task_name))
            in_progress.sort(key=lambda item: (item[0], item[1]))
            next_time = in_progress[0][0]
            time_cursor = next_time
            finished_now = sorted(
                task_name for end_time, task_name in in_progress if end_time == next_time
            )
            in_progress = [
                item for item in in_progress if item[0] != next_time
            ]
            completed.extend(finished_now)

        prompt_rows = "; ".join(
            f"{task['name']} dur={task['duration']} deps={','.join(task['deps']) if task['deps'] else '-'}"
            for task in tasks
        )
        prompt = (
            "There are two identical workers. At time 0, repeatedly start the lexicographically "
            "smallest available task on each free worker. A task becomes available only when all of "
            "its prerequisites are finished. Tasks are non-preemptive. If multiple tasks finish at "
            "the same time, record them in alphabetical order in the finish log. "
            f"Tasks: {prompt_rows}. "
            "Reply only as TOTAL|ORDER, where TOTAL is the final completion time and ORDER is the finish log."
        )
        return TestCase(
            test_id=f"dependency_schedule_extreme_{idx}",
            family="dependency_schedule_extreme",
            prompt=prompt,
            expected=f"{time_cursor}|{''.join(completed)}",
            source="dynamic",
        )

    def mk_long_trace_extreme(idx: int) -> TestCase:
        nums = rng.sample(range(2, 28), rng.randint(16, 20))
        acc = rng.randint(-12, 12)
        alt = rng.randint(-6, 6)
        flag = rng.randint(0, 4)
        bias = rng.randint(1, 4)
        mul = rng.randint(2, 5)
        sub = rng.randint(2, 6)
        start_acc = acc
        start_alt = alt
        start_flag = flag

        for i, x in enumerate(nums, start=1):
            if (x + flag + i) % 6 == 0:
                acc = acc * mul + x - flag - i
                alt = alt + bias - (x % 3)
                flag = (flag + 1) % 5
            elif x % 5 == 0:
                acc = acc - x * sub + alt + bias
                alt = alt + (i % 3) - 1
                flag = (flag + 3) % 5
            elif (i + alt) % 2 == 0:
                acc = acc + x * bias - flag + alt
            else:
                acc = acc + x + flag - alt

            if acc % 4 == 0:
                alt = alt + 1
            elif acc % 3 == 0:
                alt = alt - 1

        prompt = (
            "Without showing work, what integer does this Python function return?\n\n"
            "def run():\n"
            f"    acc = {start_acc}\n"
            f"    alt = {start_alt}\n"
            f"    flag = {start_flag}\n"
            f"    nums = {nums}\n"
            "    for i, x in enumerate(nums, start=1):\n"
            f"        if (x + flag + i) % 6 == 0:\n"
            f"            acc = acc * {mul} + x - flag - i\n"
            f"            alt = alt + {bias} - (x % 3)\n"
            "            flag = (flag + 1) % 5\n"
            "        elif x % 5 == 0:\n"
            f"            acc = acc - x * {sub} + alt + {bias}\n"
            "            alt = alt + (i % 3) - 1\n"
            "            flag = (flag + 3) % 5\n"
            "        elif (i + alt) % 2 == 0:\n"
            f"            acc = acc + x * {bias} - flag + alt\n"
            "        else:\n"
            "            acc = acc + x + flag - alt\n"
            "\n"
            "        if acc % 4 == 0:\n"
            "            alt = alt + 1\n"
            "        elif acc % 3 == 0:\n"
            "            alt = alt - 1\n"
            "    return acc\n\n"
            "Reply with only the integer."
        )
        return TestCase(
            test_id=f"long_trace_extreme_{idx}",
            family="long_trace_extreme",
            prompt=prompt,
            expected=str(acc),
            source="dynamic",
        )

    def mk_mixed_pipeline_extreme(idx: int) -> TestCase:
        teams = ["red", "blue", "green", "gold", "black"]
        regions = ["east", "west", "north", "south", "central"]
        rows = []
        for i in range(rng.randint(11, 14)):
            rows.append(
                {
                    "name": chr(ord("A") + i),
                    "team": rng.choice(teams),
                    "region": rng.choice(regions),
                    "score": rng.randint(3, 18),
                    "weight": rng.randint(1, 6),
                    "bonus": rng.randint(0, 5),
                    "penalty": rng.randint(0, 3),
                }
            )

        blocked_team = rng.choice(teams)
        blocked_region = rng.choice(regions)
        threshold = rng.randint(8, 12)
        selected = [
            row
            for row in rows
            if row["team"] != blocked_team
            and row["region"] != blocked_region
            and row["score"] >= threshold
            and row["bonus"] >= row["penalty"]
        ]
        selected.sort(
            key=lambda row: (row["weight"], -(row["score"] + row["bonus"]), row["name"])
        )
        picked = selected[:4]
        values = [row["score"] + row["bonus"] - row["penalty"] for row in picked]
        checksum = 0
        for i, value in enumerate(values, start=1):
            if i % 2 == 1:
                checksum += value * 2
            else:
                checksum -= value
        ids = "".join(row["name"] for row in picked)
        prompt_rows = "; ".join(
            f"{row['name']} team={row['team']} region={row['region']} score={row['score']} weight={row['weight']} bonus={row['bonus']} penalty={row['penalty']}"
            for row in rows
        )
        prompt = (
            "This is a two-stage task. Reply only as IDS|CHECKSUM. "
            f"Rows: {prompt_rows}. "
            f"Stage 1: keep rows where team != {blocked_team}, region != {blocked_region}, score >= {threshold}, and bonus >= penalty. "
            "Sort remaining rows by weight ascending, then (score + bonus) descending, then name ascending. "
            "Take the first 4 rows after sorting. "
            "Stage 2: for the selected rows, compute value = score + bonus - penalty. "
            "Let CHECKSUM start at 0. For values in order: add 2*value at odd positions and subtract value at even positions. "
            "Set IDS to the selected row names concatenated in order."
        )
        return TestCase(
            test_id=f"mixed_pipeline_extreme_{idx}",
            family="mixed_pipeline_extreme",
            prompt=prompt,
            expected=f"{ids}|{checksum}",
            source="dynamic",
        )

    def mk_edge_case_filter_extreme(idx: int) -> TestCase:
        statuses = ["live", "hold", "drop", "retry"]
        locks = ["yes", "no"]
        rows = []
        for i in range(rng.randint(8, 11)):
            rows.append(
                {
                    "name": chr(ord("A") + i),
                    "status": rng.choice(statuses),
                    "lock": rng.choice(locks),
                    "qty": rng.randint(5, 15),
                    "bonus": rng.randint(0, 5),
                    "penalty": rng.randint(0, 3),
                }
            )

        threshold = rng.randint(7, 12)
        selected = []
        for row in rows:
            keep = row["status"] != "drop"
            if row["status"] == "retry":
                keep = True
            if row["lock"] == "yes":
                keep = False
            if row["qty"] == threshold and row["status"] == "hold":
                keep = False
            if row["bonus"] < row["penalty"]:
                keep = False
            if keep:
                selected.append(row)

        selected.sort(
            key=lambda row: (
                -(row["qty"] + row["bonus"] - row["penalty"]),
                row["name"],
            )
        )
        picked = selected[:5]
        ids = "".join(row["name"] for row in picked)
        total = sum(row["qty"] + row["bonus"] - row["penalty"] for row in picked)
        prompt_rows = "; ".join(
            f"{row['name']} status={row['status']} lock={row['lock']} qty={row['qty']} bonus={row['bonus']} penalty={row['penalty']}"
            for row in rows
        )
        prompt = (
            "This task has boundary rules. Reply only as IDS|TOTAL. "
            f"Rows: {prompt_rows}. "
            "Rules in order: "
            "1) Default keep rows where status != drop. "
            "2) Exception: status=retry rows are kept even if rule 1 would reject them. "
            "3) Hard stop: lock=yes rows are never kept. "
            f"4) Boundary rule: if qty == {threshold} and status == hold, drop the row. "
            "5) If bonus < penalty, drop the row. "
            "After filtering, sort by (qty + bonus - penalty) descending, then name ascending. "
            "Take the first 5 rows. IDS is their names concatenated; TOTAL is sum(qty + bonus - penalty) over those rows."
        )
        return TestCase(
            test_id=f"edge_case_filter_extreme_{idx}",
            family="edge_case_filter_extreme",
            prompt=prompt,
            expected=f"{ids}|{total}",
            source="dynamic",
        )

    standard_families = [
        mk_py_trace,
        mk_path,
        mk_combinatorics,
        mk_json_extract,
        mk_transform,
        mk_digitsum,
    ]
    hard_families = [
        mk_register_machine,
        mk_table_query,
        mk_fsm,
    ]
    very_hard_families = [
        mk_register_machine_long,
        mk_table_query_hard,
        mk_fsm_long,
    ]
    extreme_families = [
        mk_register_machine_extreme,
        mk_table_query_extreme,
        mk_fsm_extreme,
        mk_stack_machine_extreme,
        mk_window_scan_extreme,
        mk_json_contract_extreme,
        mk_dependency_schedule_extreme,
        mk_long_trace_extreme,
        mk_mixed_pipeline_extreme,
        mk_edge_case_filter_extreme,
    ]
    if difficulty == "standard":
        families = standard_families
    elif difficulty == "hard":
        families = hard_families
    elif difficulty == "very-hard":
        families = very_hard_families
    elif difficulty == "extreme":
        families = extreme_families
    else:
        families = standard_families + hard_families
    for idx in range(1, tasks_per_family + 1):
        for family in families:
            tests.append(family(idx))
    return tests


def load_shadow_prompts(
    prompt_file: pathlib.Path,
    seed: int,
    sample_size: int,
) -> list[TestCase]:
    suffix = prompt_file.suffix.lower()
    raw_items: list[Any]

    if suffix == ".jsonl":
        raw_items = []
        for line in prompt_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw_items.append(json.loads(line))
    elif suffix == ".json":
        parsed = json.loads(prompt_file.read_text())
        if isinstance(parsed, list):
            raw_items = parsed
        else:
            raise ValueError(f"{prompt_file} must contain a JSON array")
    else:
        raw_items = []
        for idx, line in enumerate(prompt_file.read_text().splitlines(), start=1):
            line = line.strip()
            if line:
                raw_items.append({"id": f"line_{idx}", "prompt": line})

    cases: list[TestCase] = []
    for idx, item in enumerate(raw_items, start=1):
        if isinstance(item, str):
            prompt = item.strip()
            expected: str | None = None
            item_id = f"shadow_{idx}"
            family = "shadow_real"
            metadata: dict[str, Any] = {}
        elif isinstance(item, dict):
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            expected_value = item.get("expected")
            expected = None if expected_value is None else str(expected_value)
            item_id = str(item.get("id") or f"shadow_{idx}")
            family = str(item.get("family") or "shadow_real")
            metadata = {
                key: value
                for key, value in item.items()
                if key not in {"id", "family", "prompt", "expected"}
            }
        else:
            continue

        cases.append(
            TestCase(
                test_id=item_id,
                family=family,
                prompt=prompt,
                expected=expected,
                source="shadow",
                metadata=metadata,
            )
        )

    if sample_size > 0 and len(cases) > sample_size:
        rng = random.Random(seed)
        cases = rng.sample(cases, sample_size)
    return cases


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def extract_responses_output(data: dict[str, Any]) -> str:
    return "".join(
        chunk.get("text", "")
        for item in (data.get("output") or [])
        for chunk in (item.get("content") or [])
        if chunk.get("type") == "output_text"
    ).strip()


def list_output_item_types(data: dict[str, Any]) -> list[str]:
    return [str(item.get("type")) for item in (data.get("output") or [])]


def classify_response_issue(status: Any, output_text: str) -> str | None:
    if status == "completed" and not output_text:
        return "completed_but_no_final_answer"
    return None


def run_request(provider: ProviderConfig, prompt: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": provider.model,
        "input": prompt,
        "max_output_tokens": provider.max_output_tokens,
        "temperature": 0,
    }
    if provider.reasoning_mode == "nested":
        payload["reasoning"] = {"effort": provider.reasoning_effort}
    elif provider.reasoning_mode == "top_level":
        payload["reasoning_effort"] = provider.reasoning_effort
    if "longent.tech" in provider.url:
        payload["store"] = False

    cmd = [
        "curl",
        "-sS",
        provider.url,
        "-H",
        f"Authorization: Bearer {provider.api_key}",
        "-H",
        "Content-Type: application/json",
        "--data",
        json.dumps(payload, separators=(",", ":")),
        "--max-time",
        "90",
    ]
    if provider.needs_user_agent:
        cmd[2:2] = ["-A", "Mozilla/5.0"]

    start = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed_ms = int((time.time() - start) * 1000)
    row: dict[str, Any] = {
        "elapsed_ms": elapsed_ms,
        "returncode": res.returncode,
    }
    if res.returncode != 0:
        row["error"] = (res.stderr or res.stdout)[:300]
        return row

    try:
        data = json.loads(res.stdout)
    except Exception as exc:
        row["parse_error"] = str(exc)
        row["body_preview"] = res.stdout[:200]
        return row

    usage = data.get("usage") or {}
    completion_reasoning = (usage.get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    )
    output_reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")

    output_text = extract_responses_output(data)
    status = data.get("status")
    row.update(
        {
            "status": status,
            "returned_model": data.get("model"),
            "returned_reasoning_effort": (data.get("reasoning") or {}).get("effort"),
            "returned_service_tier": data.get("service_tier"),
            "reasoning_tokens": completion_reasoning
            if isinstance(completion_reasoning, int)
            else output_reasoning,
            "output_text": output_text,
            "output_item_types": list_output_item_types(data),
            "response_issue": classify_response_issue(status, output_text),
        }
    )
    return row


def summarize(rows: list[dict[str, Any]], provider_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name in provider_names:
        provider_rows = [r for r in rows if r["provider"] == name]
        completed = [r for r in provider_rows if r.get("status") == "completed"]
        scored = [r for r in provider_rows if r.get("expected") is not None]
        scored_completed = [r for r in completed if r.get("expected") is not None]
        issue_counts: dict[str, int] = {}
        for row in provider_rows:
            issue = row.get("response_issue")
            if isinstance(issue, str) and issue:
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        reasoning_tokens = [
            r["reasoning_tokens"]
            for r in completed
            if isinstance(r.get("reasoning_tokens"), int)
        ]

        families = sorted({r["family"] for r in provider_rows})
        family_breakdown: dict[str, Any] = {}
        for family in families:
            fam_rows = [r for r in provider_rows if r["family"] == family]
            family_issue_counts: dict[str, int] = {}
            for row in fam_rows:
                issue = row.get("response_issue")
                if isinstance(issue, str) and issue:
                    family_issue_counts[issue] = family_issue_counts.get(issue, 0) + 1
            family_breakdown[family] = {
                "completed": sum(1 for r in fam_rows if r.get("status") == "completed"),
                "correct": sum(1 for r in fam_rows if r.get("correct")),
                "scored_total": sum(1 for r in fam_rows if r.get("expected") is not None),
                "total": len(fam_rows),
                "response_issues": family_issue_counts,
            }

        summary[name] = {
            "total": len(provider_rows),
            "completed": len(completed),
            "completion_rate_pct": round(
                100 * len(completed) / len(provider_rows), 1
            )
            if provider_rows
            else None,
            "scored_total": len(scored),
            "correct": sum(1 for r in scored if r.get("correct")),
            "accuracy_scored_all_pct": round(
                100 * sum(1 for r in scored if r.get("correct")) / len(scored), 1
            )
            if scored
            else None,
            "accuracy_scored_completed_pct": round(
                100 * sum(1 for r in scored_completed if r.get("correct"))
                / len(scored_completed),
                1,
            )
            if scored_completed
            else None,
            "avg_elapsed_ms": round(
                sum(r["elapsed_ms"] for r in completed) / len(completed), 1
            )
            if completed
            else None,
            "median_reasoning_tokens": int(statistics.median(reasoning_tokens))
            if reasoning_tokens
            else None,
            "response_issues": issue_counts,
            "reasoning_efforts": sorted(
                {str(r.get("returned_reasoning_effort")) for r in completed}
            ),
            "family_breakdown": family_breakdown,
        }
    return summary


def compute_comparisons(
    rows: list[dict[str, Any]], baseline_name: str, provider_names: list[str]
) -> dict[str, Any]:
    by_provider_test = {
        provider: {row["test"]: row for row in rows if row["provider"] == provider}
        for provider in provider_names
    }
    baseline_rows = by_provider_test[baseline_name]
    comparisons: dict[str, Any] = {}

    baseline_completed = sum(
        1 for row in baseline_rows.values() if row.get("status") == "completed"
    )
    baseline_total = len(baseline_rows)
    baseline_completion_rate = (
        baseline_completed / baseline_total if baseline_total else 0.0
    )
    baseline_scored = [row for row in baseline_rows.values() if row.get("expected") is not None]
    baseline_accuracy = (
        sum(1 for row in baseline_scored if row.get("correct")) / len(baseline_scored)
        if baseline_scored
        else 0.0
    )
    baseline_reasoning = [
        row.get("reasoning_tokens")
        for row in baseline_rows.values()
        if isinstance(row.get("reasoning_tokens"), int)
    ]
    baseline_reasoning_median = (
        statistics.median(baseline_reasoning) if baseline_reasoning else None
    )

    for provider in provider_names:
        if provider == baseline_name:
            continue
        provider_rows = by_provider_test[provider]
        provider_completed = sum(
            1 for row in provider_rows.values() if row.get("status") == "completed"
        )
        provider_total = len(provider_rows)
        provider_completion_rate = (
            provider_completed / provider_total if provider_total else 0.0
        )
        provider_scored = [
            row for row in provider_rows.values() if row.get("expected") is not None
        ]
        provider_accuracy = (
            sum(1 for row in provider_scored if row.get("correct")) / len(provider_scored)
            if provider_scored
            else 0.0
        )

        comparable = []
        effort_matches = 0
        output_matches = 0
        provider_reasoning = []
        provider_latencies = []
        baseline_latencies = []
        baseline_total_rows = 0

        for test_id, base_row in baseline_rows.items():
            other_row = provider_rows.get(test_id)
            if other_row is None:
                continue
            baseline_total_rows += 1
            if (
                base_row.get("status") == "completed"
                and other_row.get("status") == "completed"
            ):
                comparable.append((base_row, other_row))
                if base_row.get("output_text") == other_row.get("output_text"):
                    output_matches += 1
                if (
                    base_row.get("returned_reasoning_effort")
                    == other_row.get("returned_reasoning_effort")
                ):
                    effort_matches += 1
                if isinstance(other_row.get("reasoning_tokens"), int):
                    provider_reasoning.append(other_row["reasoning_tokens"])
                if isinstance(base_row.get("elapsed_ms"), int):
                    baseline_latencies.append(base_row["elapsed_ms"])
                if isinstance(other_row.get("elapsed_ms"), int):
                    provider_latencies.append(other_row["elapsed_ms"])

        output_agreement = output_matches / len(comparable) if comparable else 0.0
        effort_match = effort_matches / len(comparable) if comparable else 0.0
        provider_reasoning_median = (
            statistics.median(provider_reasoning) if provider_reasoning else None
        )
        token_drift = 0.0
        if (
            baseline_reasoning_median
            and provider_reasoning_median
            and baseline_reasoning_median > 0
            and provider_reasoning_median > 0
        ):
            token_drift = min(
                1.0,
                abs(log2(provider_reasoning_median / baseline_reasoning_median)) / 2.0,
            )

        suspicion_score = min(
            100.0,
            round(
                35.0 * (1.0 - output_agreement)
                + 25.0 * abs(provider_accuracy - baseline_accuracy)
                + 20.0 * abs(provider_completion_rate - baseline_completion_rate)
                + 10.0 * (1.0 - effort_match)
                + 10.0 * token_drift,
                1,
            ),
        )

        baseline_latency_median = (
            statistics.median(baseline_latencies) if baseline_latencies else None
        )
        provider_latency_median = (
            statistics.median(provider_latencies) if provider_latencies else None
        )
        latency_ratio = (
            round(provider_latency_median / baseline_latency_median, 3)
            if baseline_latency_median and provider_latency_median
            else None
        )
        reasoning_ratio = (
            round(provider_reasoning_median / baseline_reasoning_median, 3)
            if baseline_reasoning_median and provider_reasoning_median
            else None
        )

        comparisons[provider] = {
            "baseline": baseline_name,
            "comparable_completed_pairs": len(comparable),
            "output_agreement_rate_pct": round(100 * output_agreement, 1)
            if comparable
            else None,
            "reasoning_effort_match_rate_pct": round(100 * effort_match, 1)
            if comparable
            else None,
            "completion_rate_gap_pct": round(
                100 * (provider_completion_rate - baseline_completion_rate), 1
            ),
            "accuracy_gap_pct": round(100 * (provider_accuracy - baseline_accuracy), 1),
            "median_reasoning_tokens_ratio_vs_baseline": reasoning_ratio,
            "median_latency_ratio_vs_baseline": latency_ratio,
            "suspicion_score_0_to_100": suspicion_score,
        }

    return comparisons


def build_report_rows(
    tests: list[TestCase],
    rows: list[dict[str, Any]],
    include_prompt_text: bool,
) -> list[dict[str, Any]]:
    test_lookup = {test.test_id: test for test in tests}
    output: list[dict[str, Any]] = []
    for row in rows:
        test = test_lookup[row["test"]]
        cooked = dict(row)
        cooked["source"] = test.source
        cooked["prompt_hash"] = prompt_hash(test.prompt)
        cooked["prompt_metadata"] = test.metadata
        if include_prompt_text:
            cooked["prompt"] = test.prompt
        output.append(cooked)
    return output


def main() -> int:
    args = parse_args()

    if args.tasks_per_family < 0:
        raise ValueError("--tasks-per-family must be >= 0")
    if not args.openrouter_key:
        raise ValueError("--openrouter-key is required")
    if args.max_output_tokens < 32:
        raise ValueError("--max-output-tokens must be >= 32 for xhigh runs")

    providers = [
        ProviderConfig(
            name="openrouter_responses_openai_gpt-5.4_xhigh",
            url=args.openrouter_url,
            api_key=args.openrouter_key,
            model=args.openrouter_model,
            reasoning_mode="nested",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
        ),
        ProviderConfig(
            name="longent_gpt-5.4-fast(xhigh)",
            url="https://longent.tech/v1/responses",
            api_key=load_longent_key(args.longent_config),
            model=args.longent_model,
            reasoning_mode="none",
            reasoning_effort=args.reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            needs_user_agent=True,
        ),
    ]
    if args.fireworks_key:
        providers.append(
            ProviderConfig(
                name=f"fireworks_{args.fireworks_model.split('/')[-1]}_{args.fireworks_reasoning_effort}",
                url=args.fireworks_url,
                api_key=args.fireworks_key,
                model=args.fireworks_model,
                reasoning_mode="top_level",
                reasoning_effort=args.fireworks_reasoning_effort,
                max_output_tokens=args.max_output_tokens,
            )
        )

    seed = args.seed if args.seed is not None else secrets.randbits(63)

    tests: list[TestCase] = []
    if args.tasks_per_family > 0:
        tests.extend(
            build_dynamic_test_suite(seed, args.tasks_per_family, args.difficulty)
        )
    if args.prompt_file is not None:
        tests.extend(load_shadow_prompts(args.prompt_file, seed, args.shadow_sample_size))
    if not tests:
        raise ValueError(
            "No tests to run. Provide --prompt-file and/or keep --tasks-per-family > 0."
        )

    rows: list[dict[str, Any]] = []
    for test in tests:
        for provider in providers:
            print(
                f"[audit] {provider.name} :: {test.test_id} ({test.family}/{test.source})",
                file=sys.stderr,
            )
            row = run_request(provider, test.prompt)
            row.update(
                {
                    "test": test.test_id,
                    "family": test.family,
                    "provider": provider.name,
                    "expected": test.expected,
                }
            )
            row["correct"] = (
                None
                if test.expected is None
                else row.get("output_text") == test.expected
            )
            rows.append(row)

    provider_names = [provider.name for provider in providers]
    summary = summarize(rows, provider_names)
    comparisons = compute_comparisons(rows, provider_names[0], provider_names)
    report_rows = build_report_rows(tests, rows, args.include_prompt_text)

    report = {
        "meta": {
            "seed": seed,
            "tasks_per_family": args.tasks_per_family,
            "difficulty": args.difficulty,
            "shadow_prompt_file": str(args.prompt_file) if args.prompt_file else None,
            "shadow_sample_size": args.shadow_sample_size,
            "generated_tests": len(tests),
            "providers": [
                asdict(provider) | {"api_key": "***redacted***"} for provider in providers
            ],
            "mode": "short-cycle shadow audit",
            "note": "Questions are dynamic private variants and/or private shadow prompts, not a fixed public benchmark set.",
        },
        "summary": summary,
        "comparisons": comparisons,
        "rows": report_rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "summary": summary,
                "comparisons": comparisons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
