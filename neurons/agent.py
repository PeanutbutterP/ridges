import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

# Sandbox variables
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID")
RUN_ID = os.getenv("RUN_ID") or EVALUATION_RUN_ID

def inference(model: str, temperature: float, messages: list) -> str:
    """Safe proxy inference call"""
    try:
        payload = {
            "run_id": RUN_ID or "default",
            "model": model,
            "temperature": temperature,
            "messages": messages
        }
        import requests
        resp = requests.post(
            f"{SANDBOX_PROXY_URL}/api/inference",
            json=payload,
            timeout=60
        )
        resp.raise_for_status()
        return resp.text.strip('"')
    except Exception as e:
        print(f"[SN21] Inference error: {e}")
        return "I couldn't find a definitive answer."

def get_research_context(query: str) -> str:
    """Pull real context from research_bot.log"""
    try:
        log_path = Path.home() / "bittensor-claw" / "research_bot.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
            return "".join(lines[-60:])
    except:
        pass
    return "General knowledge mode active."

def agent_main(input: Dict[str, Any]) -> str:
    """Final Optimized SN21 Search/Retrieval Agent"""
    start_time = time.time()
    query = input.get("query", "").strip()
    
    if not query:
        return json.dumps({"response": "No query provided.", "sources": []})

    print(f"[SN21] Processing query: {query}")

    # Get research context
    context = get_research_context(query)

    # Build prompt
    messages = [
        {"role": "system", "content": "You are a high-accuracy search/retrieval agent. Provide concise, factual answers with sources when possible."},
        {"role": "user", "content": f"Query: {query}\n\nRecent Research Context:\n{context}\n\nGive a clear, sourced response."}
    ]

    response = inference("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 0.3, messages)

    # Format as required JSON
    result = {
        "response": response or "I couldn't find a definitive answer.",
        "sources": ["research_bot.log", "proxy_inference"]
    }

    print(f"[SN21] Response generated in {time.time() - start_time:.1f}s")

    return json.dumps(result)

# Local test
if __name__ == "__main__":
    print("SN21 Search/Retrieval Agent loaded successfully.")
    print("Ready for testing on SN21.")
