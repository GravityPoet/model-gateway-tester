# LLM Shadow Audit

`LLM Shadow Audit` is a lightweight black-box auditing tool for comparing third-party LLM gateways against a chosen baseline using dynamic private benchmarks and short-cycle shadow evaluation.

It is designed for questions like:

- Is a gateway really serving the model it claims?
- Does a provider silently degrade quality under load?
- Which route is better for reasoning-heavy workloads?
- How do providers differ on completion rate, latency, and reasoning-token behavior?

## What It Does

- Generates dynamic private benchmark tasks by default
- Supports short-cycle shadow audit using your own prompt files
- Compares multiple providers under the same output budget
- Records completion rate, accuracy, latency, and reasoning-token distributions
- Produces a JSON report with per-family breakdowns and comparison metrics

## Current Provider Support

- OpenRouter Responses API
- Longent Responses-compatible gateway
- Fireworks Responses-compatible route

## Why This Exists

Public benchmark questions are easy for providers to overfit or recognize.

This project instead focuses on:

- Hidden dynamic tasks
- Reproducible seeds when you want them
- Repeated black-box comparisons
- Distribution-level signals rather than single anecdotal prompts

## Installation

Python 3.11+ is recommended.

No third-party Python dependencies are required.

```bash
python3 -m py_compile llm_shadow_audit.py
```

## Basic Usage

Run a default hard audit with fresh randomized questions every time:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY"
```

Add Fireworks:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --fireworks-key "YOUR_FIREWORKS_KEY"
```

Override Longent model:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --longent-model "gpt-5.4(xhigh)"
```

Reproduce the exact same benchmark set:

```bash
python3 llm_shadow_audit.py \
  --seed 123456 \
  --openrouter-key "YOUR_OPENROUTER_KEY"
```

## Shadow Audit Mode

You can feed your own real or semi-real prompts:

```bash
python3 llm_shadow_audit.py \
  --openrouter-key "YOUR_OPENROUTER_KEY" \
  --prompt-file prompts.jsonl \
  --shadow-sample-size 30 \
  --tasks-per-family 0
```

Supported prompt file formats:

- `.jsonl`
- `.json`
- `.txt`
- `.md`

Example JSONL rows:

```json
{"id":"p1","prompt":"Summarize this article"}
{"id":"p2","prompt":"Find the bug in this Python code"}
{"id":"p3","prompt":"Return strict minified JSON","expected":"{\"ok\":true}"}
```

## Default Behavior

Current defaults are tuned for stronger audits:

- `difficulty=hard`
- `tasks_per_family=10`
- `max_output_tokens=1024`
- fresh random seed per run unless `--seed` is provided

## Output

Reports are written to a timestamped JSON file by default.

The report includes:

- summary metrics per provider
- family-by-family breakdown
- row-level results
- comparison metrics against the baseline provider

Prompt text is not included unless you pass `--include-prompt-text`.

## Security Notes

- Do not commit API keys
- Prefer passing keys by CLI flag from a secure local environment
- For private real prompts, keep `--include-prompt-text` off unless you explicitly need raw prompt capture

## License

MIT
