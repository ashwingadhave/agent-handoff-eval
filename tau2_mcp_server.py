"""MCP server that serves one τ²-bench task's tools over stdio.

The agent never touches τ²-bench directly. It lists and calls the domain's
tools (including transfer_to_human_agents) through this server, and reads the
domain policy as an MCP resource. Each server process holds one fresh copy of
the task's database, so every conversation starts from the same state.

    python tau2_mcp_server.py --domain airline --task-id 13
"""

import argparse
import asyncio
import sys
import uuid

from loguru import logger
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from tau2.data_model.message import ToolCall
from tau2.registry import registry

# stdout carries the MCP protocol, so τ²-bench logs go to stderr only.
logger.remove()
logger.add(sys.stderr, level="WARNING")


def load_environment(domain: str, task_id: str):
    env = registry.get_env_constructor(domain)()
    tasks = {task.id: task for task in registry.get_tasks_loader(domain)(None)}
    task = tasks[task_id]
    state = task.initial_state
    if state is not None:
        env.set_state(
            initialization_data=state.initialization_data,
            initialization_actions=state.initialization_actions,
            message_history=state.message_history or [],
        )
    return env


def build_server(domain: str, task_id: str) -> Server:
    env = load_environment(domain, task_id)
    policy_uri = f"tau2://{domain}/policy"

    async def list_tools(ctx, params):
        tools = []
        for tool in env.get_tools():
            function = tool.openai_schema["function"]
            tools.append(
                types.Tool(
                    name=function["name"],
                    description=function.get("description", ""),
                    input_schema=function["parameters"],
                )
            )
        return types.ListToolsResult(tools=tools)

    async def call_tool(ctx, params: types.CallToolRequestParams):
        message = env.get_response(
            ToolCall(
                id=str(uuid.uuid4()),
                name=params.name,
                arguments=params.arguments or {},
                requestor="assistant",
            )
        )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=message.content)],
            is_error=message.error,
        )

    async def list_resources(ctx, params):
        return types.ListResourcesResult(
            resources=[
                types.Resource(
                    name="policy",
                    uri=policy_uri,
                    description=f"Agent policy for the τ²-bench {domain} domain.",
                    mime_type="text/markdown",
                )
            ]
        )

    async def read_resource(ctx, params: types.ReadResourceRequestParams):
        if str(params.uri) != policy_uri:
            raise ValueError(f"Unknown resource: {params.uri}")
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=policy_uri, mime_type="text/markdown", text=env.get_policy()
                )
            ]
        )

    return Server(
        f"tau2-{domain}",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
        on_list_resources=list_resources,
        on_read_resource=read_resource,
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=["airline", "retail"])
    parser.add_argument("--task-id", required=True)
    args = parser.parse_args()

    server = build_server(args.domain, args.task_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
