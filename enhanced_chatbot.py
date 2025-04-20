import os
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Union, Optional
from dotenv import load_dotenv
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain.tools.render import format_tool_to_openai_function
from langgraph.graph import END, StateGraph
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class ChatState(TypedDict):
    """Type definition for chat state."""
    messages: List[Dict[str, str]]
    current_tool: Optional[str]
    tool_result: Optional[str]

def convert_to_langchain_messages(messages: List[Dict[str, str]]) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
    """Convert dict messages to LangChain message objects."""
    message_map = {
        "user": HumanMessage,
        "assistant": AIMessage,
        "system": SystemMessage
    }
    
    langchain_messages = []
    for msg in messages:
        message_class = message_map.get(msg["role"])
        if message_class:
            langchain_messages.append(message_class(content=msg["content"]))
    
    return langchain_messages

def create_agent():
    """Create and configure the agent with tools."""
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=os.getenv("GROK_API_KEY"),
        model_name="llama3-8b-8192",
        temperature=0.7,
    )

    # Initialize tools
    arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    search_tool = TavilySearchResults(api_key=os.getenv("TAVILY_API_KEY"))

    tools = [arxiv_tool, wikipedia_tool, search_tool]
    tool_map = {tool.name: tool for tool in tools}
    
    # Convert tools to OpenAI functions format
    functions = [format_tool_to_openai_function(t) for t in tools]
    
    # Define the tool calling node
    def should_use_tool(state: Dict) -> Dict:
        """Determine if a tool should be used based on the current state."""
        messages = convert_to_langchain_messages(state["messages"])
        response = llm.predict_messages(
            messages,
            functions=functions
        )
        
        if response.additional_kwargs.get("function_call"):
            function_call = response.additional_kwargs["function_call"]
            return {
                "messages": state["messages"],
                "current_tool": function_call["name"],
                "tool_result": None,
                "__next_node__": "call_tool"
            }
        
        new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
        return {
            "messages": new_messages,
            "current_tool": None,
            "tool_result": None,
            "__next_node__": END
        }

    def call_tool(state: Dict) -> Dict:
        """Execute the specified tool."""
        messages = state["messages"]
        last_message = messages[-1]["content"]
        tool_name = state["current_tool"]
        
        if tool_name is None:
            raise ValueError("No tool specified")
            
        tool = tool_map[tool_name]
        result = tool.invoke(last_message)
        
        return {
            "messages": messages,
            "current_tool": tool_name,
            "tool_result": result,
            "__next_node__": "process_result"
        }

    def process_tool_result(state: Dict) -> Dict:
        """Process the result from the tool execution."""
        result = state["tool_result"]
        tool_name = state["current_tool"]
        messages = state["messages"]
        
        # Add tool result as system message
        new_messages = messages + [{
            "role": "system",
            "content": f"Tool {tool_name} returned: {result}"
        }]
        
        # Get AI response
        langchain_messages = convert_to_langchain_messages(new_messages)
        response = llm.predict_messages(langchain_messages)
        
        # Add AI response to messages
        final_messages = new_messages + [{"role": "assistant", "content": response.content}]
        
        return {
            "messages": final_messages,
            "current_tool": None,
            "tool_result": None,
            "__next_node__": "tool_decision"
        }

    # Create the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("tool_decision", should_use_tool)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("process_result", process_tool_result)
    
    # Set entry point
    workflow.set_entry_point("tool_decision")
    
    # Compile the graph
    chain = workflow.compile()
    
    return chain

def process_message(chain, message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Process a message through the agent chain."""
    state = {
        "messages": history + [{"role": "user", "content": message}],
        "current_tool": None,
        "tool_result": None
    }
    
    result = chain.invoke(state)
    return result["messages"] 