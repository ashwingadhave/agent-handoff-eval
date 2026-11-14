"""Run τ²-bench tasks with a Claude agent that reaches its tools over MCP.

For each task and each repeat:
  1. Start tau2_mcp_server.py for the task (fresh database every run).
  2. Read the domain policy from the server's MCP resource and list its tools.
  3. τ²-bench's user simulator plays the customer; the Claude agent answers,
     calling tools only through the MCP session.
  4. Save the conversation as TruLens conversation records
     ({"input", "tool_calls", "output"} per turn) plus the benchmark's answer
     on whether a transfer was expected.

    python run_conversations.py --repeats 5 --out runs/pilot
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import anthropic
from loguru import logger
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, get_default_environment, stdio_client
from tau2.data_model.message import AssistantMessage
from tau2.registry import registry
from tau2.user.user_simulator import STOP, TRANSFER, OUT_OF_SCOPE, UserSimulator

logger.remove()
logger.add(sys.stderr, level="WARNING")

HERE = Path(__file__).parent
TRANSFER_TOOL = "transfer_to_human_agents"
FIRST_AGENT_MESSAGE = "Hi! How can I help you today?"  # τ²-bench's default opener

# τ²-bench's own agent prompt (src/tau2/agent/llm_agent.py), so results stay
# comparable with the benchmark's reference agent.
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()
SYSTEM_PROMPT = "<instructions>\n{instruction}\n</instructions>\n<policy>\n{policy}\n</policy>"

# Tasks whose expected outcome on transfer we test (see README).
DEFAULT_TASKS = {
    "airline": ["13", "3", "5", "6", "35", "45", "47"],
    "retail": ["10", "12", "26", "50"],
}


def expects_transfer(task) -> bool:
    actions = (task.evaluation_criteria.actions or []) if task.evaluation_criteria else []
    return any(action.name == TRANSFER_TOOL for action in actions)


def strip_markers(text: str) -> str:
    for marker in (STOP, TRANSFER, OUT_OF_SCOPE):
        text = text.replace(marker, "")
    return text.strip()


async def agent_turn(client, session, args, system, tools, messages, usage):
    """Run the agent until it replies to the user. Returns (reply, tool_calls)."""
    tool_calls = []
    texts = []
    for _ in range(args.max_tool_steps):
        response = await client.messages.create(
            model=args.agent_model,
            max_tokens=16000,
            system=system,
            tools=tools,
            messages=messages,
            output_config={"effort": args.agent_effort},
            cache_control={"type": "ephemeral"},
        )
        usage["agent_input_tokens"] += response.usage.input_tokens
        usage["agent_cache_read_tokens"] += response.usage.cache_read_input_tokens or 0
        usage["agent_cache_write_tokens"] += response.usage.cache_creation_input_tokens or 0
        usage["agent_output_tokens"] += response.usage.output_tokens
        if response.stop_reason == "refusal":
            raise RuntimeError(f"agent refusal: {response.stop_details}")
        messages.append({"role": "assistant", "content": response.content})
        texts += [block.text for block in response.content if block.type == "text"]

        uses = [block for block in response.content if block.type == "tool_use"]
        if not uses:
            reply = "\n".join(t for t in texts if t.strip())
            return reply or None, tool_calls

        results = []
        for use in uses:
            out = await session.call_tool(use.name, use.input)
            text = "\n".join(c.text for c in out.content if c.type == "text")
            tool_calls.append(
                {
                    "name": use.name,
                    "arguments": use.input,
                    "result": None if out.is_error else text,
                    "error": text if out.is_error else None,
                }
            )
            results.append(
                {"type": "tool_result", "tool_use_id": use.id, "content": text, "is_error": bool(out.is_error)}
            )
        messages.append({"role": "user", "content": results})
    raise RuntimeError("agent exceeded max tool steps in one turn")


async def run_one(domain, task, repeat, args, client, out_dir):
    out_path = out_dir / f"{domain}-{task.id}-r{repeat}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())

    # The MCP client passes only a few environment variables to the server;
    # keep PYTHONPATH so the server finds τ²-bench when it is set.
    env = get_default_environment()
    if os.environ.get("PYTHONPATH"):
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(HERE / "tau2_mcp_server.py"), "--domain", domain, "--task-id", task.id],
        env=env,
    )
    usage = {
        "agent_input_tokens": 0,
        "agent_cache_read_tokens": 0,
        "agent_cache_write_tokens": 0,
        "agent_output_tokens": 0,
        "user_cost_usd": 0.0,
    }
    turns, ended_by = [], "max_turns"
    started = time.time()

    async with stdio_client(server, errlog=open("/dev/null", "w")) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resources = (await session.list_resources()).resources
            policy = (await session.read_resource(str(resources[0].uri))).contents[0].text
            tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in (await session.list_tools()).tools
            ]
            system = SYSTEM_PROMPT.format(instruction=AGENT_INSTRUCTION, policy=policy)

            user = UserSimulator(llm=args.user_model, instructions=str(task.user_scenario))
            user_state = user.get_init_state()
            agent_text = FIRST_AGENT_MESSAGE
            messages = []

            for _ in range(args.max_turns):
                user_msg, user_state = await asyncio.to_thread(
                    user.generate_next_message,
                    AssistantMessage(role="assistant", content=agent_text or "..."),
                    user_state,
                )
                usage["user_cost_usd"] += user_msg.cost or 0.0
                if UserSimulator.is_stop(user_msg):
                    ended_by = "user_stop"
                    leftover = strip_markers(user_msg.content or "")
                    if leftover:
                        turns.append({"input": leftover, "output": None})
                    break
                messages.append({"role": "user", "content": user_msg.content})
                agent_text, tool_calls = await agent_turn(
                    client, session, args, system, tools, messages, usage
                )
                turn = {"input": user_msg.content, "output": agent_text}
                if tool_calls:
                    turn["tool_calls"] = tool_calls
                turns.append(turn)

    transferred = any(
        call["name"] == TRANSFER_TOOL and call["error"] is None
        for turn in turns
        for call in turn.get("tool_calls", [])
    )
    result = {
        "domain": domain,
        "task_id": task.id,
        "repeat": repeat,
        "agent_model": args.agent_model,
        "agent_effort": args.agent_effort,
        "user_model": args.user_model,
        "expected_transfer": expects_transfer(task),
        "transferred": transferred,
        "ended_by": ended_by,
        "policy": policy,
        "seconds": round(time.time() - started, 1),
        "usage": usage,
        "turns": turns,
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/pilot")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--tasks", help="e.g. airline:13,retail:10 (default: the 11 test tasks)")
    parser.add_argument("--agent-model", default="claude-opus-5-5")
    parser.add_argument("--agent-effort", default="medium")
    parser.add_argument("--user-model", default="anthropic/claude-opus-5-5")
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-tool-steps", type=int, default=15)
    parser.add_argument("--parallel", type=int, default=3)
    args = parser.parse_args()

    wanted = DEFAULT_TASKS
    if args.tasks:
        wanted = {}
        for item in args.tasks.split(","):
            domain, task_id = item.split(":")
            wanted.setdefault(domain, []).append(task_id)

    out_dir = HERE / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic()
    limit = asyncio.Semaphore(args.parallel)

    jobs = []
    for domain, ids in wanted.items():
        tasks = {t.id: t for t in registry.get_tasks_loader(domain)(None)}
        for task_id in ids:
            for repeat in range(1, args.repeats + 1):
                jobs.append((domain, tasks[task_id], repeat))

    async def guarded(job):
        async with limit:
            domain, task, repeat = job
            try:
                r = await run_one(domain, task, repeat, args, client, out_dir)
                print(
                    f"{domain}-{task.id} r{repeat}: expected_transfer={r['expected_transfer']} "
                    f"transferred={r['transferred']} turns={len(r['turns'])} {r['seconds']}s",
                    flush=True,
                )
            except Exception as exc:
                print(f"{domain}-{task.id} r{repeat}: ERROR {type(exc).__name__}: {exc}", flush=True)

    await asyncio.gather(*(guarded(job) for job in jobs))


if __name__ == "__main__":
    asyncio.run(main())
