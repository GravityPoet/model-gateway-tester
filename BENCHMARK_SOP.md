# Benchmark SOP

## Goal

Run fresh, fair, reproducible reasoning benchmarks across multiple OpenAI-compatible routes.

## Fairness Rules

Use the same:

- test set
- seed
- difficulty
- `reasoning.effort`
- `max_output_tokens`
- timeout budget
- scoring rule

Do not compare runs if one model gets a larger `max_output_tokens` than the others unless the report is explicitly labeled as a configuration-tuning run rather than a fair benchmark.

## Public Default

For a public repository, the default benchmark should assume that others can inspect:

- the generator logic
- the family structure
- the scoring rules
- past result files

So the public default should be:

- `difficulty=extreme`
- fresh random seed
- the largest current built-in family mix
- enough questions to avoid single-family domination

In this repo, that means:

- default main benchmark = fresh 30-question `extreme` run

Treat low-difficulty or tiny runs as `smoke` / `debug`, not as the headline benchmark.

## Current Suite Shape

The built-in `extreme` generator now mixes ten structured-but-dynamic families:

- `register_machine_extreme`
- `table_query_extreme`
- `fsm_extreme`
- `stack_machine_extreme`
- `window_scan_extreme`
- `json_contract_extreme`
- `dependency_schedule_extreme`
- `long_trace_extreme`
- `mixed_pipeline_extreme`
- `edge_case_filter_extreme`

This is materially better than the old three-family setup, but it is still a synthetic reasoning suite rather than a general intelligence benchmark.

## Standard Commands

### Five-route fair run

Use `ai558669_head2head.py` with 2-5 provider entries and unified `max_output_tokens`.

### Longent budget sweep

Use `longent_budget_sweep.py` to hold the test set fixed and vary only `max_output_tokens`.

### Mixed-provider fair run

Use `fiveway_extreme20.py` when comparing multiple providers under one shared fresh suite.

### Smoke / Debug run

Use a reduced command only for connectivity checks, schema validation, or fast route triage:

```bash
python3 model_gateway_tester.py \
  --difficulty hard \
  --tasks-per-family 1 \
  --max-output-tokens 512
```

## Reporting Rules

Always report:

- completed / total
- correct / total
- `accuracy_scored_all_pct`
- `accuracy_scored_completed_pct`
- `avg_elapsed_ms`
- `median_reasoning_tokens`
- `response_issues`

Call out `completed_but_no_final_answer` separately from normal wrong answers.

## Interpretation Rules

If two models have the same correctness:

- prefer higher completion rate
- then prefer lower latency

If a route loses mainly on completion:

- describe it as a stability problem, not a reasoning-quality problem

If a route returns `completed` with no `output_text`:

- describe it as output-budget or route-behavior failure, not as an incorrect answer

## Next Upgrade Path

To improve separation even further, add more dynamic families beyond the current ten:

- longer context row filtering
- multi-family mixed reasoning tasks
- stricter nested JSON/schema contracts
- stronger adversarial instruction traps
- hybrid tasks that combine 3+ substeps

Do that in [model_gateway_tester.py](/Users/moonlitpoet/Tools/model-gateway-tester/model_gateway_tester.py), then reuse the existing harness scripts without changing the scoring pipeline.
