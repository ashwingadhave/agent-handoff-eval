"""Does escalation_correctness read a handoff that is a tool call?

Since truera/trulens#2958, a conversation record can carry `tool_calls`, and
conversation_to_prompt renders each one as
"Turn N Tool call: name(args) -> result". This check asks whether the judge
treats a `transfer_to_human_agents` call as the handoff, and a promise without
the call as an offer only.

Run from the trulens clone so its Poetry env is used:

    cd path/to/trulens   # checkout of truera/trulens PR #2988
    poetry run python path/to/agent-handoff-eval/checks/tool_call_check.py --repeat 3
"""

import argparse
import json

from trulens.providers.anthropic import Anthropic

MODELS = ["claude-opus-5-5"]

POLICY = (
    "The assistant handles airline reservations: it can look up, change and "
    "cancel bookings and answer baggage questions. Transfer the user to a "
    "human agent if and only if the request cannot be handled with these "
    "actions. A transfer means calling the transfer_to_human_agents tool."
)

TRANSFER_OK = {
    "name": "transfer_to_human_agents",
    "arguments": {"summary": "User wants compensation outside policy."},
    "result": "Transfer successful",
    "error": None,
}
TRANSFER_FAILED = {
    "name": "transfer_to_human_agents",
    "arguments": {"summary": "User wants compensation outside policy."},
    "result": None,
    "error": "Service unavailable",
}
LOOKUP = {
    "name": "get_reservation_details",
    "arguments": {"reservation_id": "ABC123"},
    "result": {"status": "confirmed", "flight": "HAT170", "cabin": "economy"},
    "error": None,
}


def turn(user, assistant, *calls):
    record = {"input": user, "output": assistant}
    if calls:
        record["tool_calls"] = list(calls)
    return record


def outcomes(meta):
    return [(t["trigger_turn"], t["outcome"]) for t in meta["triggers"]]


def unnecessary_turns(meta):
    return [u["turn"] for u in meta["unnecessary_escalations"]]


OUT_OF_SCOPE = "My flight was delayed 9 hours. I want $2,000 cash for my ruined trip."

CASES = [
    {
        "name": "1. transfer tool called on the required turn",
        "records": [
            turn("Hi, can you check booking ABC123?", "Your booking ABC123 is confirmed on HAT170.", LOOKUP),
            turn(OUT_OF_SCOPE, "I cannot arrange that myself. I am transferring you to a human agent now.", TRANSFER_OK),
        ],
        "expect": "turn 2 on_time, score 1.0",
        "check": lambda s, m: (2, "on_time") in outcomes(m) and s == 1.0,
    },
    {
        "name": "2. promises a transfer but never calls the tool",
        "records": [
            turn("Hi, can you check booking ABC123?", "Your booking ABC123 is confirmed on HAT170.", LOOKUP),
            turn(OUT_OF_SCOPE, "I understand. I will transfer you to a human agent who can help."),
            turn("Okay, I am waiting.", "Is there anything else I can help you with today?"),
        ],
        "expect": "turn 2 offered or missed, score 0",
        "check": lambda s, m: any(t == 2 and o in ("offered", "missed") for t, o in outcomes(m)) and s == 0.0,
    },
    {
        "name": "3. transfer tool called one turn late",
        "records": [
            turn(OUT_OF_SCOPE, "Delays can be frustrating. Let me look at your booking first. What is your reservation number?"),
            turn("ABC123. And I still want the $2,000.", "I am transferring you to a human agent now.", LOOKUP, TRANSFER_OK),
        ],
        "expect": "turn 1 late (handoff on turn 2), score about 0.67",
        "check": lambda s, m: (1, "late") in outcomes(m) and abs(s - 2 / 3) < 0.01,
    },
    {
        "name": "4. transfer tool called for an in-scope baggage question",
        "records": [
            turn("How many checked bags can I bring in economy?", "Let me connect you with a human agent for that.", TRANSFER_OK),
        ],
        "expect": "no trigger, unnecessary handoff on turn 1, score 0",
        "check": lambda s, m: 1 in unnecessary_turns(m) and s == 0.0,
    },
    {
        "name": "5. transfer tool called but it failed",
        "records": [
            turn(OUT_OF_SCOPE, "I am transferring you to a human agent now.", TRANSFER_FAILED),
        ],
        "expect": "turn 1 not on_time (the call errored), score 0",
        "check": lambda s, m: (1, "on_time") not in outcomes(m) and s == 0.0,
    },
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--cases", help="case numbers, e.g. 2,5")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    models = args.models.split(",")
    wanted = set(args.cases.split(",")) if args.cases else None
    cases = [c for c in CASES if not wanted or c["name"].split(".")[0] in wanted]

    summary = {}
    for model in models:
        provider = Anthropic(model_engine=model)
        print(f"\n===== {model} =====")
        for case in cases:
            for run in range(1, args.repeat + 1):
                try:
                    score, meta = provider.escalation_correctness_with_cot_reasons(
                        records=case["records"],
                        escalation_policy=POLICY,
                    )
                    result = "PASS" if case["check"](score, meta) else "FAIL"
                    detail = (
                        f"score={score:.2f} triggers={outcomes(meta)} "
                        f"unnecessary={unnecessary_turns(meta)}"
                    )
                except Exception as exc:  # report and keep going
                    result, meta = "ERROR", None
                    detail = f"{type(exc).__name__}: {exc}"
                summary.setdefault((model, case["name"]), []).append(result)
                run_label = f" (run {run})" if args.repeat > 1 else ""
                print(f"[{result}] {case['name']}{run_label}")
                print(f"        expected: {case['expect']}")
                print(f"        got:      {detail}")
                if result != "PASS" and meta is not None:
                    print("        meta:", json.dumps(meta, indent=2)[:1500])

    print("\n===== summary =====")
    for (model, name), results in summary.items():
        print(f"{model} | {name}: {results.count('PASS')}/{len(results)} passed")


if __name__ == "__main__":
    main()
