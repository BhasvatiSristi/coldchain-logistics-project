import os
import sys

from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# ==========================================
# 1. SETUP & PATH RESOLUTION
# ==========================================

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[0]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")


# ==========================================
# 2. IMPORT AGENT TOOLS
# ==========================================

from src.agent_tools import (
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop
)


# ==========================================
# 3. AGENT STATE
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================
# 4. OLLAMA LLM INITIALIZATION
# ==========================================

AGENT_LLM_SETTING = os.getenv(
    "Agent_llm",
    "OLLAMA"
).strip().upper()


if AGENT_LLM_SETTING != "OLLAMA":
    raise ValueError(
        f"Unsupported Agent_llm setting: {AGENT_LLM_SETTING}. "
        f"Please set Agent_llm=OLLAMA in your .env"
    )


print("🧠 Brain Mode: Local Ollama (Mistral 7B)...")


from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)


# ==========================================
# 5. BIND TOOLS
# ==========================================

fde_tools = [
    query_telemetry_db,
    fetch_corridor_conditions,
    search_compliance_sop
]


print("🔧 Binding FDE tools to the local LLM...")


llm_with_tools = llm.bind_tools(fde_tools)


# ==========================================
# 6. REASONING NODE
# ==========================================

def reasoning_node(state: AgentState):

    response = llm_with_tools.invoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# ==========================================
# 7. GRAPH ARCHITECTURE
# ==========================================

print("⚙️ Compiling LangGraph FDE Orchestrator...")


graph_builder = StateGraph(AgentState)


# Reasoning node
graph_builder.add_node(
    "reasoner",
    reasoning_node
)


# Tool execution node
graph_builder.add_node(
    "tools",
    ToolNode(fde_tools)
)


# START → Reasoner
graph_builder.add_edge(
    START,
    "reasoner"
)


# Reasoner → Tools OR END
graph_builder.add_conditional_edges(
    "reasoner",
    tools_condition
)


# Tools → Reasoner
graph_builder.add_edge(
    "tools",
    "reasoner"
)


# Compile graph with memory
fde_agent = graph_builder.compile(
    checkpointer=MemorySaver()
)


# ==========================================
# 8. CHAT LOOP TESTING PANEL
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print(
        "🚀 FDE Supply Chain Orchestrator State Machine Online"
    )

    print(
        f"   Configured Execution: "
        f"[LLM: {AGENT_LLM_SETTING}]"
    )

    print(
        "   Model: Mistral 7B via Ollama"
    )

    print("=" * 60 + "\n")


    # --------------------------------------
    # Load system prompt
    # --------------------------------------

    prompt_path = (
        project_root
        / "src"
        / "prompts"
        / "system_prompt.txt"
    )


    try:

        with open(
            prompt_path,
            "r",
            encoding="utf-8"
        ) as f:

            system_instructions = f.read()


    except FileNotFoundError:

        print(
            f"⚠️ Warning: Could not find {prompt_path}"
        )

        system_instructions = (
            "You are a helpful AI assistant."
        )


    system_prompt = SystemMessage(
        content=system_instructions
    )


    # --------------------------------------
    # Conversation thread
    # --------------------------------------

    thread_config = {
        "configurable": {
            "thread_id": "production_test_1"
        }
    }


    # --------------------------------------
    # Initialize conversation
    # --------------------------------------

    fde_agent.invoke(
        {
            "messages": [
                system_prompt
            ]
        },
        config=thread_config
    )


    # --------------------------------------
    # Interactive loop
    # --------------------------------------

    while True:

        user_input = input(
            "\nDispatcher > "
        )


        if user_input.lower() in [
            "exit",
            "quit"
        ]:

            print("\n👋 FDE Agent shutting down...")

            break


        print("\n⏳ Processing...")


        try:

            events = fde_agent.stream(
                {
                    "messages": [
                        ("user", user_input)
                    ]
                },
                config=thread_config,
                stream_mode="updates"
            )


            # ----------------------------------
            # Process LangGraph events
            # ----------------------------------

            for event in events:

                for node_name, node_state in event.items():

                    # ------------------------------
                    # TOOL EXECUTION
                    # ------------------------------

                    if node_name == "tools":

                        print(
                            "   [System] 🔄 "
                            "Retrieving external data "
                            "via ToolNode..."
                        )


                    # ------------------------------
                    # REASONER
                    # ------------------------------

                    elif node_name == "reasoner":

                        latest_msg = (
                            node_state["messages"][-1]
                        )


                        if latest_msg.content:

                            print(
                                f"\n🤖 FDE Agent:\n"
                                f"{latest_msg.content}"
                            )


        except Exception as e:

            print(
                "\n❌ Agent Error:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )

            print(
                "\nPlease check that Ollama is running "
                "and the Mistral model is installed."
            )