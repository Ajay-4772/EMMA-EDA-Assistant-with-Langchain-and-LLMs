import requests

def query_llm(prompt: str, model: str = "mistral", api_type: str = "ollama") -> str:
    """
    Query a local LLM (Ollama) for data analysis assistance
    
    Args:
        prompt: The user's query
        model: The model name to use (default: mistral)
        api_type: Type of API (default: ollama)
    
    Returns:
        The LLM's response as a string
    """
    if api_type == "ollama":
        return query_ollama(prompt, model)
    else:
        return f"❌ Unsupported API type: {api_type}"

def query_ollama(prompt: str, model: str = "mistral") -> str:
    """Query Ollama API for data analysis assistance"""
    url = "http://localhost:11434/api/generate"
    
    # Enhanced conversational prompt
    enhanced_prompt = f"""You are EMMA, a friendly and knowledgeable data analysis assistant. Be conversational, helpful, and engaging like ChatGPT.

{prompt}

Remember to:
- Be conversational and friendly
- Use emojis when appropriate
- Provide detailed, helpful responses
- Reference previous context if it's a follow-up question
- Generate SQL queries when asked
- Give specific insights and actionable information"""
    
    payload = {
        "model": model,
        "prompt": enhanced_prompt,
        "stream": False,
        "options": {
            "num_predict": 300,  # Increased for more detailed responses
            "temperature": 0.3,  # Slightly higher for more conversational tone
            "top_k": 20,         # More token variety
            "top_p": 0.9         # Nucleus sampling
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=15)  # Increased timeout
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "[No response from model]")
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return "⏰ Sorry, I'm taking a bit longer than expected. Please try again!"
    except Exception as e:
        return f"❌ Oops! Something went wrong: {str(e)}" 