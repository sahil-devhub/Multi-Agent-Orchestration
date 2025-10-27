import os
import io
import json
import contextlib
import re
from typing import TypedDict, List, Literal, Optional
from datetime import datetime # Make sure this import is at the top

from dotenv import load_dotenv

# LangChain / LangGraph imports
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq 

# Local Tool Import
from .tools import ALL_TOOLS, tavily_search
from langchain_community.tools.tavily_search import TavilySearchResults


# --- Environment Setup ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path)


# --- Tool Definitions ---
TOOLS = ALL_TOOLS

# Fallback map for deprecated models
DEPRECATED_MODEL_MAP = {
    "llama3-8b-8192": "llama-3.1-8b-instant",
    "llama3-70b-8192": "llama-3.1-70b-versatile",
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile" 
}

# --- LLM Factory ---
def make_llm(model_name: str, provider: str):
    """Creates an LLM instance based on the selected provider."""
    
    if model_name in DEPRECATED_MODEL_MAP:
        print(f"---WARNING: Model '{model_name}' is deprecated. Switching to '{DEPRECATED_MODEL_MAP[model_name]}'---")
        model_name = DEPRECATED_MODEL_MAP[model_name]

    if provider.upper() != "GROQ":
        raise ValueError(f"Unknown model provider: {provider}. This app is configured for GROQ only.")

    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in .env. Cannot use Groq models.")
    return ChatGroq(model=model_name, temperature=0)

# --- Graph State ---
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    persona: str
    selected_model_name: str
    selected_model_provider: str


# --- NEW NODE: Model Router ---
def model_router_node(state: AgentState) -> dict:
    """
    Dynamically selects the best Groq model based on the user's input and intended agent.
    """
    print("---ROUTING MODEL---")
    last_message = state["messages"][-1]
    has_image = False
    
    if isinstance(last_message.content, list):
        for part in last_message.content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                has_image = True
                break

    persona = state["persona"]
    
    chosen_model_name = None
    chosen_model_provider = "GROQ" 

    if has_image:
        chosen_model_name = "meta-llama/Llama-4-scout-17b-16e-instruct"
        print(f"---MODEL DECISION: Vision Task -> Groq '{chosen_model_name}'---")
            
    elif persona == "Financial Analyst":
        chosen_model_name = "llama-3.3-70b-versatile" 
        print(f"---MODEL DECISION: Financial Analyst -> Groq '{chosen_model_name}'---")

    elif persona == "Creative Writer":
        chosen_model_name = "llama-3.1-8b-instant" 
        print(f"---MODEL DECISION: Creative Writer -> Groq '{chosen_model_name}'---")
            
    if not chosen_model_name:
        raise ValueError("Could not determine a suitable model for the request.")

    return {
        "selected_model_name": chosen_model_name,
        "selected_model_provider": chosen_model_provider
    }


# --- Old Router Node (now only determines persona) ---
def persona_router_node(state: AgentState) -> dict:
    """
    Determines the correct agent persona.
    """
    print("---ROUTING PERSONA---")
    last_message = state["messages"][-1]
    query = ""
    has_image = False

    if isinstance(last_message.content, list):
        for part in last_message.content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    query = part.get("text", "").lower()
                elif part.get("type") == "image_url":
                    has_image = True
    elif isinstance(last_message.content, str):
        query = last_message.content.lower()

    financial_keywords = [
        "stock", "market", "finance", "news", "companies", "top 5", "top 10",
        "price", "bitcoin", "crypto", "investment", "rate", "usd", "inr",
        "business", "economic", "latest", "largest", "ranking"
    ]
    
    if has_image:
        decision = "Vision Agent"
    elif any(k in query for k in financial_keywords):
        decision = "Financial Analyst"
    else: 
        decision = "Creative Writer"
        
    print(f"---PERSONA DECISION: {decision}---")
    return {"persona": decision}


def agent_node(state: AgentState) -> dict:
    """
    This is the main worker node. It invokes the LLM with the current state
    and system prompt.
    """
    
    llm = make_llm(state["selected_model_name"], state["selected_model_provider"])
    persona = state["persona"]
    print(f"---ENTERING AGENT: {persona}---")

    if persona == "Vision Agent":
        print("---INVOKING VISION AGENT (NO SYSTEM PROMPT)---")
        ai_response = llm.invoke(state["messages"])
        return {"messages": [ai_response]}

    # --- THIS IS THE FIX ---
    # New prompt with conditional timestamp logic.
    if persona == "Financial Analyst":
        # Get the current time on the server.
        now_str = datetime.now().strftime("%I:%M %p on %B %d, %Y") 

        system_prompt = (
            f"You are a professional Data Analyst. You MUST use the tavily_search tool. "
            f"Your job is to provide factual, reliable, and beautifully formatted answers based *only* on the search results."
            f"\n\n"
            "**CRITICAL RULES:**"
            "\n"
            "1.  **NO HALLUCINATION:** You MUST NOT invent data. You must extract prices, numbers, and company names *directly* from the search results. DO NOT invent links or data."
            "\n"
            "2.  **SOURCE QUALITY:** You MUST prioritize authoritative sources (e.g., 'Forbes', 'Wikipedia', 'CoinMarketCap', 'Coinbase') from the search URLs."
            "\n"
            "3.  **EXPLAIN CONFLICTS:** For volatile assets (like crypto/stocks), you must report the different prices you find and add a simple, one-sentence explanation of *why* they are different."
            "\n"
            "4.  **PROFESSIONAL FORMATTING:** Your response MUST be in two sections: 'The Answer' and 'Source Links', separated by a horizontal line (`---`). Use **bolding** for key items."
            "\n"
            "5.  **CONDITIONAL TIMESTAMP:** You MUST add a timestamp (e.g., 'As of 12:01 PM on October 26, 2025, ...') **ONLY** for queries about volatile, real-time data like stock prices or cryptocurrency. Do **NOT** add a timestamp for static lists like 'Top 5 companies'."
            "\n"
            "6.  **NUMBER FORMATTING:** All prices in INR (Indian Rupees) MUST be formatted with Indian comma separators (e.g., ₹1,01,20,088.16)."
            "\n\n"
            "--- EXAMPLE 1: Static Ranking (e.g., 'Top 5 companies') ---"
            "\n"
            "**The Answer:**"
            "\n"
            "Based on recent market cap data from authoritative sources, the top 5 IT companies in India are:"
            "\n"
            "1. **Tata Consultancy Services (TCS)**"
            "\n"
            "2. **Infosys**"
            "\n"
            "3. **HCL Technologies**"
            "\n"
            "4. **Wipro**"
            "\n"
            "5. **LTIMindtree**"
            "\n\n"
            "---"
            "\n"
            "**Source Links:**"
            "\n"
            "- [Forbes India: Top IT companies in India](https://www.forbesindia.com/article/explainers/top-10-it-companies-in-india/87143/1)"
            "\n"
            "- [CompaniesMarketCap: Largest IT Service Companies](https://companiesmarketcap.com/inr/it-services/largest-it-service-companies-by-market-cap/)"
            "\n"
            "--- EXAMPLE 2: Volatile Price (e.g., 'Price of Bitcoin in INR') ---"
            "\n"
            "**The Answer:**"
            "\n"
            f"As of {now_str}, the price of Bitcoin in INR varies slightly across different exchanges. Here are the current prices as found in the search results:"
            "\n"
            "- **On Mudrex:** ₹1,01,20,088.16"
            "\n"
            "- **On CoinMarketCap:** ₹97,99,606.42"
            "\n"
            "- **On CoinSwitch:** Price not found in search snippet."
            "\n\n"
            "*Note: Prices vary by exchange based on their specific order books and trading volume.*"
            "\n\n"
            "---"
            "\n"
            "**Source Links:**"
            "\n"
            "- [Mudrex: BTC to INR](https://mudrex.com/converter/btc/inr)"
            "\n"
            "- [CoinMarketCap: BTC to INR](https://coinmarketcap.com/currencies/bitcoin/btc/inr/)"
            "\n"
            "- [CoinSwitch: BTC/INR Price](https://coinswitch.co/pro/btc-inr/csx)"
            "\n"
            "--- END OF EXAMPLES ---"
            "\n\n"
            "Your task is to populate the correct template. The tool results are a list of dictionaries like `{'url': '...', 'content': '...'}`. "
            "You must extract the `url` and `content` to build your answer. You MUST use the *real* URLs from the tool output."
        )
        llm_with_tools = llm.bind_tools([tavily_search])
    elif persona == "Creative Writer":
        system_prompt = "You are a helpful creative writing assistant."
        llm_with_tools = llm 
    else:
        system_prompt = "You are a helpful assistant."
        llm_with_tools = llm

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    ai_response = llm_with_tools.invoke(messages)
    
    return {"messages": [ai_response]}

def should_continue(state: AgentState) -> Literal["tools", "__end__", "agent"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("---DECISION: USE TOOLS---")
        return "tools"
    
    if isinstance(last_message, ToolMessage):
        print("---DECISION: RETURN TO AGENT FOR SYNTHESIS---")
        return "agent"
    
    print("---DECISION: END---")
    return "__end__"

# --- Main Public API Function ---
def get_response(
    system_prompt: str, 
    messages: List[str], 
    allow_search: bool,
    image_data: Optional[str] = None
) -> str:
    print("---INVOKING MULTI-AGENT GRAPH---")
    
    graph = StateGraph(AgentState)
    
    graph.add_node("persona_router", persona_router_node)
    graph.add_node("model_router", model_router_node) 
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    
    graph.add_edge(START, "persona_router")
    graph.add_edge("persona_router", "model_router") 
    graph.add_edge("model_router", "agent")
    
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "agent": "agent", 
            "__end__": END
        }
    )
    graph.add_edge("tools", "agent") 
    
    app = graph.compile()

    human_messages = [HumanMessage(content=m) for m in messages]

    if image_data and human_messages:
        last_message = human_messages[-1]
        multimodal_content = [
            {"type": "text", "text": last_message.content},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
            }
        ]
        human_messages[-1] = HumanMessage(content=multimodal_content)

    initial_state = {"messages": human_messages}
    final_state = app.invoke(initial_state)

    try:
        last_message = final_state["messages"][-1]
        if hasattr(last_message, 'content'):
            return str(last_message.content)
        return str(last_message)
    except (KeyError, IndexError) as e:
        print(f"Error extracting final response: {e}")
        return "Sorry, I encountered an issue processing the final response."