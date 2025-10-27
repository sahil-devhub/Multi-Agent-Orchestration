import io
import json
import contextlib
import re
from typing import List
from langchain_core.tools import tool

# Import the correct tool
from langchain_community.tools.tavily_search import TavilySearchResults


def clean_snippet(text: str) -> str:
    """
    Cleans raw search snippets by removing junk characters, newlines,
    and fixing jumbled words (like '496USD' or 'witha24').
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove newlines, tabs, and common markdown
    text = re.sub(r'[\n\r\t\*\_\|#]', ' ', text)
    
    # 2. Add a space between numbers and letters (e.g., "496USD" -> "496 USD")
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
    
    # 3. Add a space between letters and numbers (e.g., "witha24" -> "with a 24")
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    
    # 4. Remove stray artifacts like '[...]'
    text = re.sub(r'\[.*?\]', '', text)
    
    # 5. Collapse multiple spaces into one
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


@tool("tavily_search")
def tavily_search(query: str) -> List[dict]:
    """
    Performs a web search using Tavily and returns a list of result dictionaries.
    """
    try:
        tavily = TavilySearchResults(max_results=5)
        raw_results = tavily.invoke(query)

        if not raw_results:
            return [{"error": "The web search returned no results for that query. Please try rephrasing it."}]
        
        # We loop through and clean the content of each snippet
        cleaned_results = []
        for result in raw_results:
            cleaned_content = clean_snippet(result.get("content"))
            cleaned_results.append({
                "url": result.get("url"),
                "content": cleaned_content
            })
            
        return cleaned_results
        
    except Exception as e:
        return [{"error": f"An error occurred during the search: {e}"}]

# --- A list to export all tools ---
ALL_TOOLS = [tavily_search]