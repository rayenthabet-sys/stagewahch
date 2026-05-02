import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import json
from gatekeeper import gatekeeper_agent
from librarian import librarian_agent
from researcher import researcher_agent
from constructor import constructor_agent

def orchestrator(user_query: str):
    print(f"\n--- Processing Query: '{user_query}' ---\n")
    
    # 1. Gatekeeper Agent
    print("🛡️ Gatekeeper Agent: Evaluating query...")
    is_relevant = gatekeeper_agent(user_query)
    
    if not is_relevant:
        result = {
            "status": "rejected",
            "message_darija": "Sma7 lia, ana mkhases ghir f lfilaha dial zitoun. Maçedarch njaobek 3la hadchi."
        }
        return json.dumps(result, indent=2, ensure_ascii=False)
        
    print("✅ Gatekeeper passed! Query is relevant.\n")
    
    # 2. Librarian Agent (RAG)
    vector_context = librarian_agent(user_query)
    
    # 3. Researcher Agent
    search_context = researcher_agent(user_query)
    
    # 4. Constructor Agent
    final_response_json = constructor_agent(user_query, vector_context, search_context)
    
    return final_response_json

if __name__ == "__main__":
    # Test cases
    test_queries = [
        "What type of soil is best for planting olive trees?",
        "Who won the world cup in 2022?"
    ]
    
    for q in test_queries:
        result = orchestrator(q)
        print(f"\nFinal Output:\n{result}\n")
        print("="*60)
