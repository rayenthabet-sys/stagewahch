import asyncio

MOCK_KB = """
[El Maktaba - Tunisian Olive PDF Data]:
- Best varieties in Tunisia are Chemlali in the south (drought resistant) and Chetoui in the north.
- Harvesting typically starts in November when color turns purple/black.
- Common diseases: Olive fruit fly (Dacus oleae) and Peacock spot (Spilocaea oleagina).
- Watering requirement: Usually rain-fed, but supplementary irrigation in summer boosts yield by 30%.
- Pruning (Zebra) should clear the center of the tree to allow wind and sunlight to enter.
"""

async def agent_2_koutbi(query: str) -> str:
    """Agent 2 (Librarian): Mock RAG"""
    await asyncio.sleep(0.5) # simulate Vector DB latency
    if any(word in query.lower() for word in ["zaytoun", "olive", "chajra", "zitoun", "saba", "harvest", "disease", "pruning", "taks", "weather", "traba", "soil"]):
        return MOCK_KB
    return "No specific internal data found in PDFs."
