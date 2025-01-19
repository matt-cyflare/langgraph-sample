from typing import Annotated
import os
import json
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
import uuid

load_dotenv()

# Set your API key
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Initialize the memory saver
memory = MemorySaver()

# Define the state
class State(TypedDict):
    messages: Annotated[list, add_messages]

# Create the graph
graph_builder = StateGraph(State)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")
llm_with_tools = llm.bind_tools(tools)

# Define the chatbot function
def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

# Replace BasicToolNode with the prebuilt ToolNode
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

# Add conditional edges using the prebuilt tools_condition
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

# Compile the graph with memory checkpointer
graph = graph_builder.compile(checkpointer=memory)

# Create the chat loop
def stream_graph_updates(user_input: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    message = HumanMessage(content=user_input)
    
    for event in graph.stream(
        {"messages": [message]},
        config,
        stream_mode="values"
    ):
        if event:
            messages = event.get("messages", [])
            if messages and len(messages) > 0:
                # Only print if it's an AI message
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    print("Assistant:", last_message.content)

def main():
    print("Chat started! Type 'quit', 'exit', or 'q' to end the conversation.")
    thread_id = str(uuid.uuid4())
    print(f"Chat session ID: {thread_id}")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input, thread_id)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()