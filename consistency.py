"""Measure handoff consistency across repeated runs with MLflow and pass^k.

Follows the "Measure consistency with repeated runs" recipe in the MLflow
docs (genai/eval-monitor/running-evaluation/eval-examples): one parent run,
one nested run per repeat with mlflow.genai.evaluate, then pass^k per test
case from the combined result_df. The conversations already exist, so each
repeat evaluates the saved conversations for that repeat instead of calling
the agent again.

Two scorers:
  - benchmark_handoff: the agent called transfer_to_human_agents exactly
    when τ²-bench expects it.
  - escalation_correctness: the TruLens score from score.py is 1.0. Runs not
    scored yet are logged as scorer errors and skipped in pass^k.

    python consistency.py runs/main
    mlflow ui --backend-store-uri sqlite:///mlflow.db   # to browse the runs
"""

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow.entities import Feedback  # noqa: E402
from mlflow.genai.scorers import scorer  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
SCORERS = ["benchmark_handoff", "escalation_correctness"]


@scorer(name="benchmark_handoff")
def benchmark_handoff(outputs, expectations):
    ok = outputs["transferred"] == expectations["expected_transfer"]
    return Feedback(
        value="yes" if ok else "no",
        rationale=f"transferred={outputs['transferred']}, expected={expectations['expected_transfer']}",
    )


@scorer(name="escalation_correctness")
def escalation_correctness(outputs):
    score = outputs.get("trulens_score")
    if score is None:
        return Feedback(error="not scored by TruLens yet (run score.py)")
    return Feedback(
        value="yes" if score == 1.0 else "no",
        rationale=f"TruLens escalation correctness = {score:.2f}; {outputs['trulens_outcomes']}",
    )


def load_rows(runs_dir: Path) -> dict[int, list[dict]]:
    """Group saved conversations by repeat number, as evaluate() rows."""
    by_repeat: dict[int, list[dict]] = {}
    for path in sorted(runs_dir.glob("*-r*.json")):
        run = json.loads(path.read_text())
        scored = runs_dir / "scores" / path.name
        trulens = json.loads(scored.read_text()) if scored.exists() else None
        outcomes = None
        if trulens:
            meta = trulens["meta"]
            outcomes = {
                "triggers": [(t["trigger_turn"], t["outcome"]) for t in meta["triggers"]],
                "unnecessary": [u["turn"] for u in meta["unnecessary_escalations"]],
            }
        by_repeat.setdefault(run["repeat"], []).append(
            {
                "inputs": {"task": f"{run['domain']}-{run['task_id']}"},
                "outputs": {
                    "transferred": run["transferred"],
                    "turns": len(run["turns"]),
                    "trulens_score": trulens["metric_score"] if trulens else None,
                    "trulens_outcomes": outcomes,
                },
                "expectations": {"expected_transfer": run["expected_transfer"]},
            }
        )
    return by_repeat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs_dir")
    parser.add_argument("--experiment", default="agent-handoff-eval")
    args = parser.parse_args()
    runs_dir = (HERE / args.runs_dir).resolve()

    mlflow.set_experiment(args.experiment)
    by_repeat = load_rows(runs_dir)

    frames = []
    with mlflow.start_run(run_name=f"consistency-{runs_dir.name}"):
        for repeat in sorted(by_repeat):
            with mlflow.start_run(run_name=f"repeat-{repeat}", nested=True):
                result = mlflow.genai.evaluate(
                    data=by_repeat[repeat],
                    scorers=[benchmark_handoff, escalation_correctness],
                )
            frames.append(result.result_df)
        results = pd.concat(frames, ignore_index=True)

        # Each request is {"task": ...}; group on the task name. `n` counts
        # scored rows, skipping errors.
        results["case"] = results["request"].map(
            lambda r: (r if isinstance(r, dict) else json.loads(r))["task"]
        )
        max_n = results.groupby("case").size().max()
        tables = []
        for name in SCORERS:
            per_case = results.groupby("case")[f"{name}/value"].agg(
                n="count", c=lambda v: (v == "yes").sum()
            )
            for k in range(1, max_n + 1):
                per_case[f"pass^{k}"] = [
                    math.comb(c, k) / math.comb(n, k) if n >= k else float("nan")
                    for n, c in zip(per_case["n"], per_case["c"])
                ]
                mean = per_case[f"pass^{k}"].mean()
                if not math.isnan(mean):
                    # MLflow metric names cannot contain "^".
                    mlflow.log_metric(f"{name}/pass_hat_{k}", mean)
            per_case.insert(0, "scorer", name)
            tables.append(per_case.reset_index())
        table = pd.concat(tables, ignore_index=True)
        mlflow.log_table(table, artifact_file="pass_k_per_task.json")

    pd.set_option("display.width", 160)
    for name in SCORERS:
        part = table[table["scorer"] == name].drop(columns="scorer")
        if part["n"].sum() == 0:
            print(f"\n{name}: no scored runs yet")
            continue
        print(f"\n{name}")
        print(part.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
