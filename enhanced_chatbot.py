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
from langchain_core.utils.function_calling import convert_to_openai_function
from langgraph.graph import END, StateGraph
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Debug print for API keys
print("GROK_API_KEY present:", bool(os.getenv("GROK_API_KEY")))
print("TAVILY_API_KEY present:", bool(os.getenv("TAVILY_API_KEY")))

SYSTEM_PROMPT = """You are a helpful AI assistant with access to multiple tools:
1. Arxiv: For searching academic papers and research
2. Wikipedia: For general knowledge and concepts
3. Tavily Search: For current information and real-time searches

When a user asks a question:
1. First analyze if you need external information
2. If yes, choose the most appropriate tool(s)
3. Use the tool results to provide a comprehensive answer
4. Always be clear and informative

Remember to:
- Use Arxiv for academic/research questions
- Use Wikipedia for general knowledge and concepts
- Use Tavily Search for current events and real-time information
- Combine information from multiple tools when needed

If you don't need to use any tools, just provide a direct response."""

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
    # Add system prompt first
    langchain_messages.append(SystemMessage(content=SYSTEM_PROMPT))
    
    for msg in messages:
        message_class = message_map.get(msg["role"])
        if message_class:
            langchain_messages.append(message_class(content=msg["content"]))
    
    return langchain_messages

def create_agent():
    """Create and configure the agent with tools."""
    
    print("Initializing agent...")
    
    # Initialize LLM
    llm = ChatGroq(
        api_key=os.getenv("GROK_API_KEY"),
        model_name="mixtral-8x7b-32768",  # Using Mixtral model instead
        temperature=0.7,
        max_tokens=4096,
    )
    
    print("LLM initialized")

    # Initialize tools with better descriptions
    arxiv_tool = ArxivQueryRun(
        api_wrapper=ArxivAPIWrapper(),
        name="arxiv",
        description="Use this tool for searching academic papers and research articles. Input should be a search query."
    )
    
    wikipedia_tool = WikipediaQueryRun(
        api_wrapper=WikipediaAPIWrapper(),
        name="wikipedia",
        description="Use this tool for looking up general knowledge, concepts, and historical information. Input should be a search query."
    )
    
    search_tool = TavilySearchResults(
        api_key=os.getenv("TAVILY_API_KEY"),
        name="tavily_search",
        description="Use this tool for searching current events, recent information, and real-time data. Input should be a search query."
    )

    tools = [arxiv_tool, wikipedia_tool, search_tool]
    tool_map = {tool.name: tool for tool in tools}
    
    print("Tools initialized")
    
    # Convert tools to OpenAI functions format using the new method
    functions = [convert_to_openai_function(t) for t in tools]
    
    # Define the tool calling node
    def should_use_tool(state: Dict) -> Dict:
        """Determine if a tool should be used based on the current state."""
        print("Entering should_use_tool")
        messages = convert_to_langchain_messages(state["messages"])
        print(f"Messages prepared: {len(messages)} messages")
        
        try:
            print("Calling LLM for tool decision")
            response = llm.predict_messages(
                messages,
                functions=functions
            )
            print(f"LLM response received: {response}")
            
            if response.additional_kwargs.get("function_call"):
                function_call = response.additional_kwargs["function_call"]
                print(f"Tool selected: {function_call['name']}")
                return {
                    "messages": state["messages"],
                    "current_tool": function_call["name"],
                    "tool_result": None,
                    "__next_node__": "call_tool"
                }
            
            print("No tool needed, providing direct response")
            new_messages = state["messages"] + [{"role": "assistant", "content": response.content}]
            return {
                "messages": new_messages,
                "current_tool": None,
                "tool_result": None,
                "__next_node__": END
            }
        except Exception as e:
            print(f"Error in should_use_tool: {str(e)}")
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": f"An error occurred while processing your request: {str(e)}. Please try again."}],
                "current_tool": None,
                "tool_result": None,
                "__next_node__": END
            }

    def call_tool(state: Dict) -> Dict:
        """Execute the specified tool."""
        print("Entering call_tool")
        try:
            messages = state["messages"]
            last_message = messages[-1]["content"]
            tool_name = state["current_tool"]
            
            print(f"Calling tool: {tool_name}")
            print(f"With input: {last_message}")
            
            if tool_name is None:
                raise ValueError("No tool specified")
                
            tool = tool_map[tool_name]
            result = tool.invoke(last_message)
            print(f"Tool result received: {result[:100]}...")  # Print first 100 chars
            
            # Add a system message about tool usage
            tool_message = {"role": "system", "content": f"Tool {tool_name} returned: {str(result)}"}
            return {
                "messages": messages + [tool_message],
                "current_tool": tool_name,
                "tool_result": str(result),
                "__next_node__": "process_result"
            }
        except Exception as e:
            print(f"Error in call_tool: {str(e)}")
            error_msg = f"An error occurred while using the {state['current_tool']} tool: {str(e)}"
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": error_msg}],
                "current_tool": None,
                "tool_result": None,
                "__next_node__": END
            }

    def process_tool_result(state: Dict) -> Dict:
        """Process the result from the tool execution."""
        print("Entering process_tool_result")
        try:
            messages = state["messages"]
            
            # Get AI response
            print("Getting AI response to tool result")
            langchain_messages = convert_to_langchain_messages(messages)
            response = llm.predict_messages(langchain_messages)
            print(f"AI response received: {response.content[:100]}...")  # Print first 100 chars
            
            # Add AI response to messages
            final_messages = messages + [{"role": "assistant", "content": response.content}]
            
            return {
                "messages": final_messages,
                "current_tool": None,
                "tool_result": None,
                "__next_node__": END
            }
        except Exception as e:
            print(f"Error in process_tool_result: {str(e)}")
            return {
                "messages": state["messages"] + [{"role": "assistant", "content": "I apologize, but I encountered an error processing the tool results. Could you please try asking your question differently?"}],
                "current_tool": None,
                "tool_result": None,
                "__next_node__": END
            }

    print("Creating graph")
    # Create the graph
    workflow = StateGraph(ChatState)
    
    # Add nodes
    workflow.add_node("tool_decision", should_use_tool)
    workflow.add_node("call_tool", call_tool)
    workflow.add_node("process_result", process_tool_result)
    
    # Set entry point
    workflow.set_entry_point("tool_decision")
    
    # Add edges
    workflow.add_edge("tool_decision", "call_tool")
    workflow.add_edge("call_tool", "process_result")
    workflow.add_edge("process_result", END)
    
    print("Compiling graph")
    # Compile the graph
    chain = workflow.compile()
    
    print("Agent creation complete")
    return chain

def process_message(chain, message: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Process a message through the agent chain."""
    print(f"\nProcessing new message: {message}")
    try:
        state = {
            "messages": history + [{"role": "user", "content": message}],
            "current_tool": None,
            "tool_result": None
        }
        
        print("Invoking chain")
        result = chain.invoke(state)
        print("Chain execution complete")
        return result["messages"]
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "I apologize, but I encountered an error processing your request. Please try again."}
        ] 