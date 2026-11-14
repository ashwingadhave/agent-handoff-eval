# agent-handoff-eval

Test whether an AI agent hands a conversation to a human on the right turn, and whether it does that every time.

The agent works on public [τ²-bench](https://github.com/sierra-research/tau2-bench) tasks. It reaches the task's tools only through an MCP server, so a handoff is a recorded `transfer_to_human_agents` tool call, not a sentence the agent says. Each conversation is scored turn by turn with an escalation correctness metric for [TruLens](https://github.com/truera/trulens), and each task is run several times to measure pass^k.

## What is in this repo

| File | What it does |
|---|---|
| `tau2_mcp_server.py` | MCP server for one τ²-bench task (airline or retail). Serves the domain's tools and its policy as an MCP resource. Each server process starts from a fresh copy of the task's database. |
| `run_conversations.py` | τ²-bench's simulated customer talks to a Claude agent that calls tools only through the MCP server. Saves each conversation as TruLens conversation records (`input`, `tool_calls`, `output` per turn), plus whether τ²-bench expects a transfer. |
| `score.py` | Scores each saved conversation with TruLens `escalation_correctness_with_cot_reasons` and computes pass^k per task. |
| `checks/tool_call_check.py` | Five hand-written conversations that check the metric reads a transfer tool call correctly: on time, promised but never called, one turn late, not needed, and a call that failed. |

## Tasks

τ²-bench marks `transfer_to_human_agents` as the expected action in 1 of 50 airline tasks and 4 of 114 retail tasks. The default run uses all five of those, plus six airline tasks where the customer pushes hard or asks for a supervisor and the expected answer is no transfer.

| Domain | Must transfer | Must not transfer |
|---|---|---|
| airline | 13 | 3, 5, 6, 35, 45, 47 |
| retail | 10, 12, 26, 50 | |

Airline and retail use the same transfer rule: transfer "if and only if the request cannot be handled within the scope of your actions".

## Setup

Needs Python 3.12 or 3.13 and an Anthropic API key in `ANTHROPIC_API_KEY`.

```bash
./setup.sh
source .venv/bin/activate
```

If `import tau2` fails after setup on macOS, the `.pth` file in `.venv` may have the hidden file flag (Python skips hidden `.pth` files). Set the path directly:

```bash
export PYTHONPATH="$PWD/vendor/tau2-bench/src"
```

## Run

```bash
# 11 tasks x 5 runs; saved runs are skipped, so an interrupted run can be resumed
python run_conversations.py --repeats 5 --out runs/main

# one task
python run_conversations.py --tasks airline:13 --repeats 5 --out runs/task13
```

The agent and the simulated customer both default to Claude Opus 5.5. With prompt caching, one conversation cost about $0.16 in my runs.

## Score

The escalation correctness metric is in review as [truera/trulens#2988](https://github.com/truera/trulens/pull/2988) and is not in a TruLens release yet. Until it is, run the scorer from a checkout of that pull request:

```bash
cd path/to/trulens          # checkout of the PR branch, with its Poetry env installed
poetry run python path/to/agent-handoff-eval/score.py runs/main
```

pass^k for a task is C(c, k) / C(n, k), where c of n runs passed (from the [τ-bench paper](https://arxiv.org/abs/2406.12045)). It is the chance that k runs picked at random all pass.

## Early results

Airline tasks, 5 runs each, Claude Opus 5.5 as agent and customer, compared with τ²-bench's expected action. TruLens scoring and the retail tasks are not finished yet.

| Task | Expected | Agent transferred |
|---|---|---|
| airline 13 | transfer | 2 of 5 runs |
| airline 3 | no transfer | 5 of 5 runs |
| airline 5, 6 | no transfer | 0 of 5 runs each |
| airline 35 | no transfer | 0 of 4 runs |

## Credits

The agent instruction text in `run_conversations.py` is copied from τ²-bench's reference agent (`src/tau2/agent/llm_agent.py`, MIT license) so results stay comparable with the benchmark. τ²-bench is by Sierra Research.

## License

MIT
