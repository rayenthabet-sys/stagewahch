import asyncio
import requests
from config import TAVILY_API_KEY

async def agent_3_ba7at(query: str) -> str:
    """Agent 3 (Researcher): Tavily Web Search"""
    def _search():
        if not TAVILY_API_KEY:
            return "No Tavily API key."
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY, 
                    "query": query + " Tunisia olive farming recent news or weather", 
                    "search_depth": "basic", 
                    "include_answer": True
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            return data.get("answer", "No relevant external data found.")
        except requests.exceptions.Timeout:
            return "Search API timed out."
        except Exception as e:
            return f"Search API error: {str(e)}"
            
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _search)
    except Exception as e:
        return f"Async Ba7at Error: {e}"
