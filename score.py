"""Score saved conversations with TruLens escalation correctness, then pass^k.

Needs TruLens with the escalation correctness metric (truera/trulens PR #2988).
Run it from a TruLens checkout of that branch so its Poetry env is used:

    cd path/to/trulens
    poetry run python path/to/agent-handoff-eval/score.py runs/main

Each conversation gets two verdicts:
  - metric: the escalation correctness score is 1.0 (every required handoff on
    time, no unnecessary handoff), judged against the full domain policy.
  - benchmark: the agent called transfer_to_human_agents exactly when the
    τ²-bench task expects it.

pass^k per task = C(c, k) / C(n, k), where c of n repeats passed
(τ-bench, arXiv 2406.12045). It is the chance that k runs picked at random
all pass.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from math import comb
from pathlib import Path

from trulens.providers.anthropic import Anthropic

HERE = Path(__file__).parent


def pass_hat_k(c: int, n: int, k: int) -> float:
    return comb(c, k) / comb(n, k) if n >= k else float("nan")


def score_run(provider, run_path: Path, out_dir: Path) -> dict:
    out_path = out_dir / run_path.name
    if out_path.exists():
        return json.loads(out_path.read_text())
    run = json.loads(run_path.read_text())
    # The judge gets the same full domain policy the agent was given.
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

    by_task = defaultdict(list)
    for r in results:
        by_task[(r["domain"], r["task_id"])].append(r)

    print(f"{'task':<12} {'must?':<6} {'n':>2} {'xfer':>4}  {'bench c':>7} {'pass^1':>6} {'pass^n':>6}  "
          f"{'metric c':>8} {'pass^1':>6} {'pass^n':>6}  agree")
    summary = []
    for (domain, task_id), rs in sorted(by_task.items()):
        n = len(rs)
        bench_c = sum(r["benchmark_pass"] for r in rs)
        metric_c = sum(r["metric_pass"] for r in rs)
        agree = sum(r["benchmark_pass"] == r["metric_pass"] for r in rs)
        row = {
            "task": f"{domain}-{task_id}",
            "must_transfer": rs[0]["expected_transfer"],
            "n": n,
            "transferred": sum(r["transferred"] for r in rs),
            "bench_c": bench_c,
            "bench_pass1": pass_hat_k(bench_c, n, 1),
            "bench_passn": pass_hat_k(bench_c, n, n),
            "metric_c": metric_c,
            "metric_pass1": pass_hat_k(metric_c, n, 1),
            "metric_passn": pass_hat_k(metric_c, n, n),
            "agree": agree,
        }
        summary.append(row)
        print(f"{row['task']:<12} {str(row['must_transfer']):<6} {n:>2} {row['transferred']:>4}  "
              f"{bench_c:>7} {row['bench_pass1']:>6.2f} {row['bench_passn']:>6.2f}  "
              f"{metric_c:>8} {row['metric_pass1']:>6.2f} {row['metric_passn']:>6.2f}  {agree}/{n}")

    for label, group in (("must transfer", True), ("must not transfer", False)):
        rows = [r for r in summary if r["must_transfer"] is group]
        if not rows:
            continue
        for kind in ("bench", "metric"):
            p1 = sum(r[f"{kind}_pass1"] for r in rows) / len(rows)
            pn = sum(r[f"{kind}_passn"] for r in rows) / len(rows)
            print(f"{label:<18} {kind:<6}: mean pass^1={p1:.2f}  mean pass^n={pn:.2f}  "
                  f"(tasks all-pass: {sum(r[f'{kind}_c'] == r['n'] for r in rows)}/{len(rows)})")
    total = sum(r["agree"] for r in summary)
    print(f"metric vs benchmark agreement: {total}/{sum(r['n'] for r in summary)} conversations")
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
