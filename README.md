# Model Gateway Tester

Compare model gateways the simple way.

`Model Gateway Tester` helps you test whether different AI providers and third-party routes are:

- strong enough
- stable enough
- fast enough
- really acting like the model they claim to serve

It is built for people who want a practical answer to questions like:

- "Is this third-party GPT route actually good?"
- "Does this provider get worse under pressure?"
- "Which route should I use for reasoning-heavy work?"
- "Why does one gateway feel smarter than another?"

## What This Project Does

- generates fresh private benchmark questions by default
- supports short-cycle shadow audit with your own prompt samples
- compares providers under the same output budget
- tracks completion rate, accuracy, latency, and reasoning-token behavior
- writes a machine-readable JSON report you can inspect or compare later

## Why It Is Useful

Public benchmark questions are easy for providers to overfit, recognize, or route around.

This project focuses on tougher-to-fake evaluation:

- private dynamic tasks
- reproducible runs when you want them
- side-by-side route comparisons
- output behavior, not just one lucky answer

## Features

- Dynamic private benchmark generation
- Hard-mode task families for reasoning-heavy checks
- Shadow audit mode using your own prompts
- Support for OpenRouter, Longent, and Fireworks-style routes
- Fresh random seed by default on every run
- Optional fixed seed for exact reproduction

## Supported Input Modes

1. Dynamic audit
This generates fresh hidden questions every run.

2. Shadow audit
This uses your own prompts from:

- `.jsonl`
- `.json`
- `.txt`
- `.md`

## Quick Start

Python 3.11+ is recommended.

```bash
python3 -m py_compile llm_shadow_audit.py
```

Run a default hard audit:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY"
```

Run a multi-provider comparison:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --fireworks-key "YOUR_FIREWORKS_KEY"
```

Override the Longent route:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --longent-model "gpt-5.4(xhigh)"
```

Reproduce a previous run exactly:

```bash
python3 llm_shadow_audit.py \
  --seed 123456 \
  --openrouter-key "YOUR_OPENROUTER_KEY"
```

## Shadow Audit Example

Use your own prompt sample file:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --prompt-file prompts.jsonl \
  --shadow-sample-size 30 \
  --tasks-per-family 0
```

Example `jsonl` lines:

```json
{"id":"p1","prompt":"Summarize this article"}
{"id":"p2","prompt":"Find the bug in this Python code"}
{"id":"p3","prompt":"Return strict minified JSON","expected":"{\"ok\":true}"}
```

## Defaults

The current defaults are tuned for stronger audits:

- `difficulty=hard`
- `tasks_per_family=10`
- `max_output_tokens=1024`
- fresh random seed every run unless you pass `--seed`

## Output Report

Each run writes a JSON report.

The report includes:

- per-provider summary
- per-family breakdown
- row-level results
- comparison metrics versus the baseline provider

Prompt text is excluded by default. Only prompt hashes are stored unless you explicitly enable prompt capture.

## Security Notes

- never commit API keys
- pass keys via CLI flags or your own secure local setup
- keep `--include-prompt-text` off unless you truly need raw prompt storage

## Chinese Summary

`Model Gateway Tester` 是一个用来比较模型网关质量的工具。

它的作用是：

- 测第三方模型路线强不强
- 测它稳不稳
- 测它快不快
- 测它到底像不像它宣称的那个模型

它不是模型本身，而是一个“模型网关测试器”。

## License

MIT
