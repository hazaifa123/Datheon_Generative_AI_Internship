import os
import ast
import operator
import sys
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ---------------------------------------------------------
# 1. Define Custom & Built-in Tools
# ---------------------------------------------------------
ddg_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """Search the web in real time using DuckDuckGo for current facts, news, or information."""
    try:
        return ddg_search.invoke(query)
    except Exception as e:
        return f"Web search error: {e}"


@tool
def math_calculator(expression: str) -> str:
    """Safely evaluate mathematical arithmetic expressions (e.g., '25 * 48 + 120 / 4' or '2 ** 10')."""
    allowed_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.left), _eval(node.right))
        elif isinstance(node, ast.UnaryOp) and type(node.op) in allowed_operators:
            return allowed_operators[type(node.op)](_eval(node.operand))
        else:
            raise ValueError("Unsupported mathematical expression.")

    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval(tree.body)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error for '{expression}': {e}"


@tool
def read_local_file(file_path: str) -> str:
    """Read and return the text content of a local file (e.g., 'notes.txt', 'data.md')."""
    try:
        safe_path = os.path.abspath(file_path)
        cwd = os.path.abspath(os.getcwd())
        if not safe_path.startswith(cwd):
            return "Access denied: Can only read files within the project directory."

        if not os.path.exists(safe_path):
            return f"File not found: '{file_path}'"

        with open(safe_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content if content else "[File is empty]"
    except Exception as e:
        return f"Error reading file '{file_path}': {e}"


# ---------------------------------------------------------
# 2. Initialize Autonomous ReAct Agent with Reasoning Trace
# ---------------------------------------------------------
def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0
    )

    tools = [web_search, math_calculator, read_local_file]

    system_message = (
        "You are an autonomous multi-tool AI research and problem-solving agent. "
        "Break down compound user queries into logical steps, invoke the appropriate tools "
        "(web_search, math_calculator, read_local_file), verify observations, and synthesize "
        "a clear, accurate final answer."
    )

    agent = create_react_agent(llm, tools=tools, prompt=system_message)
    return agent


def run_agent_with_trace(agent, query: str):
    """Streams and prints the Thought -> Action -> Observation -> Final Answer execution loop."""
    print("\n--- Agent Reasoning & Execution Chain ---")
    step_num = 1
    final_response = ""

    for chunk in agent.stream({"messages": [("user", query)]}, stream_mode="updates"):
        for node_name, node_output in chunk.items():
            messages = node_output.get("messages", [])
            for msg in messages:
                if node_name == "agent" and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        print(f"[Step {step_num}] THOUGHT & ACTION -> Calling tool: '{tc['name']}'")
                        print(f"         Input Arguments: {tc['args']}")
                        step_num += 1

                elif node_name == "tools":
                    obs_preview = str(msg.content).strip()
                    if len(obs_preview) > 300:
                        obs_preview = obs_preview[:300] + "... [truncated]"
                    print(f"[Step {step_num}] OBSERVATION ({msg.name}) -> {obs_preview}")
                    step_num += 1

                elif node_name == "agent" and msg.content:
                    final_response = msg.content

    print("-----------------------------------------")
    print(f"\nFINAL ANSWER:\n{final_response}\n")


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found. Please set it in your .env file or environment.")
        sys.exit(1)

    print("============================================================")
    print("   DATHEON TASK 3: MULTI-TOOL AUTONOMOUS AI AGENT (REACT)   ")
    print("============================================================")
    print("Available Tools: [web_search] | [math_calculator] | [read_local_file]")
    print("Sample File Ready: 'sample_notes.txt' (try asking the agent to read & calculate from it!)")
    print("Type 'exit' to quit.\n")

    agent = build_agent()

    while True:
        query = input("User Query: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            print("Shutting down agent.")
            break

        try:
            run_agent_with_trace(agent, query)
        except Exception as e:
            print(f"Agent execution error: {e}\n")


if __name__ == "__main__":
    main()
