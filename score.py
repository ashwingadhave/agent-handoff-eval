"""Score saved conversations with the TruLens escalation correctness metric.

Needs TruLens with the escalation correctness metric (truera/trulens PR #2988).
Run it from a TruLens checkout of that branch so its Poetry env is used:

    cd path/to/trulens
    poetry run python path/to/agent-handoff-eval/score.py runs/main

Writes one file per conversation to <runs_dir>/scores/. consistency.py then
reads them and computes pass^k in MLflow.

A conversation passes when the score is 1.0: every required handoff happened
on time and no handoff happened that the policy did not require. The judge is
given the same full domain policy the agent was given.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from trulens.providers.anthropic import Anthropic

HERE = Path(__file__).parent


def score_run(provider, run_path: Path, out_dir: Path) -> dict:
    out_path = out_dir / run_path.name
    if out_path.exists():
        return json.loads(out_path.read_text())
    run = json.loads(run_path.read_text())
    score, meta = provider.escalation_correctness_with_cot_reasons(
        records=run["turns"], escalation_policy=run["policy"]
    )
    result = {
        "domain": run["domain"],
        "task_id": run["task_id"],
        "repeat": run["repeat"],
        "expected_transfer": run["expected_transfer"],
        "transferred": run["transferred"],
        "benchmark_pass": run["transferred"] == run["expected_transfer"],
        "metric_score": score,
        "metric_pass": score == 1.0,
        "meta": meta,
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    parser.add_argument("--judge-model", default="claude-opus-5-5")
    parser.add_argument("--parallel", type=int, default=4)
    args = parser.parse_args()

    runs_dir = (HERE / args.runs_dir).resolve()
    out_dir = runs_dir / "scores"
    out_dir.mkdir(exist_ok=True)
    provider = Anthropic(model_engine=args.judge_model)

    run_paths = sorted(runs_dir.glob("*-r*.json"))
    with ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(lambda p: score_run(provider, p, out_dir), run_paths))

    # Quick check: how often does the metric agree with τ²-bench's expected action?
    by_task = defaultdict(list)
    for r in results:
        by_task[f"{r['domain']}-{r['task_id']}"].append(r)
    print(f"{'task':<12} {'n':>2} {'metric pass':>11} {'benchmark pass':>14} {'agree':>6}")
    for task, rs in sorted(by_task.items()):
        print(
            f"{task:<12} {len(rs):>2} {sum(r['metric_pass'] for r in rs):>11} "
            f"{sum(r['benchmark_pass'] for r in rs):>14} "
            f"{sum(r['metric_pass'] == r['benchmark_pass'] for r in rs):>5}/{len(rs)}"
        )
    print("Next: python consistency.py", args.runs_dir)


if __name__ == "__main__":
    main()
