from typing import (
    Annotated,
    Sequence,
    TypedDict,
    Literal,
)
import tempfile
import webbrowser
from dotenv import load_dotenv
from pathlib import Path


from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


from agent_platform.config import config_holder
from agent_platform.core.secrets import Secrets
from agent_platform.services.model import LLM

project_root = Path(__file__).parent.parent.parent.parent
load_dotenv(project_root / ".env", override=False)


class AgentState(TypedDict):
    """The state of the agent."""

    # add_messages is a reducer
    # See https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers
    messages: Annotated[Sequence[BaseMessage], add_messages]


class SimpleAgent:
    def __init__(self, tools):
        """
        Agent prototype

        :param tools: list of tools (can be @tool or any)
        """
        self.config = config_holder.get_config()
        self.secrets = Secrets(self.config.secrets.path)
        self.llm = LLM()
        self.tools = tools
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.agent = None

        self.llm.client = self.llm.client.bind_tools(self.tools)

    def tool_node(self, state: AgentState) -> Command[Literal["llm_call"]]:
        """Performs the tool call

        :param state: state of the agent
        :return langgraph command with goto mapping
        """
        result = []
        try:
            for tool_call in state["messages"][-1].tool_calls:
                tool = self.tools_by_name[tool_call["name"]]
                observation = tool.invoke(tool_call["args"])
                result.append(
                    ToolMessage(content=observation, tool_call_id=tool_call["id"])
                )
                return Command(update={"messages": result}, goto="llm_call")
        except Exception as e:
            # Let the LLM see what went wrong and try again
            return Command(
                update={"messages": f"Tool error: {str(e)}"}, goto="llm_call"
            )

    def llm_call(
        self, state: AgentState, config: RunnableConfig
    ) -> Command[Literal["tool_node", END]]:
        """
        Simple LLM call

        :param state: agent state
        :param config: internal runnable for langgraph
        :return: langgraph command with goto mapping
        """
        # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
        system_prompt = SystemMessage(
            "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
        )
        response = self.llm.client.invoke([system_prompt] + state["messages"], config)
        # We return a list, because this will get added to the existing list
        if response.tool_calls:
            goto = "tool_node"
        else:
            goto = END

        return Command(update={"messages": [response]}, goto=goto)

    def build_agent(self) -> None:
        """
        Base agent builder method with graph compiling

        :return: None
        """
        # Build workflow
        agent_builder = StateGraph(AgentState)

        # Add nodes
        agent_builder.add_node("llm_call", self.llm_call)
        agent_builder.add_node("tool_node", self.tool_node)

        # Add edges to connect nodes
        agent_builder.add_edge(START, "llm_call")
        agent_builder.add_edge("llm_call", "tool_node")
        agent_builder.add_edge("llm_call", END)

        # Compile the agent
        self.agent = agent_builder.compile()

    def view_agent_graph(self) -> None:
        """
        A browser visualization for agent graph

        :return: None
        """
        # Show the agent
        mermaid_code = self.agent.get_graph(xray=True).draw_mermaid()
        html_content = f"""<!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Agent Graph</title>
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
            </script>
        </head>
        <body>
            <pre class="mermaid">
        {mermaid_code}
            </pre>
        </body>
        </html>
        """
        # Сохраняем во временный файл
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as f:
            f.write(html_content)
            temp_path = f.name

        # Открываем в браузере
        webbrowser.open("file://" + temp_path)

    def invoke_agent(self, content="Add 3 and 4.") -> None:
        """
        Base invoke method with tracing in terminal

        :param content: any user prompt
        :return: None
        """
        # Invoke
        messages = [HumanMessage(content=content)]
        messages = self.agent.invoke({"messages": messages})
        for m in messages["messages"]:
            m.pretty_print()
