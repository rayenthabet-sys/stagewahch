from config import groq_client

async def agent_1_3assas(query: str) -> tuple[bool, str]:
    """Agent 1 (Gatekeeper): Checks if query is about agriculture and dynamically handles off-topic queries."""
    prompt = f"""
    You are '3assas', a friendly gatekeeper for a Tunisian agricultural app.
    Your job is to determine if the user's query is remotely related to agriculture, farming, crops, livestock, weather, nature, soil, or olive trees.
    You should be very loose and forgiving—if it's even slightly related to nature or farming, let it pass.
    
    If it IS related:
    Reply with exactly the word: VALID
    
    If it is completely UNRELATED:
    Generate a polite, friendly apology in Tunisian Darija (using Latin letters / Arabizi) explaining that you are a dedicated agricultural AI assistant and you can only answer questions about farming, plants, weather, animals, or agriculture.
    Be natural and friendly. Do NOT include the word 'VALID' in your response.
    
    User Query: {query}
    """
    try:
        response = await groq_client.chat.completions.create(
            model='llama-3.3-70b-versatile',
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        
        # Check if the AI approved it
        if "VALID" in text.upper()[:20]:
            return True, ""
        else:
            return False, text
    except Exception as e:
        print("3assas Error:", e)
        return True, "" # Default to allow if service is unstable
