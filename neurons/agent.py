import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

# Sandbox environment variables (from official template)
RUN_ID = os.getenv("RUN_ID")
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL")

if not RUN_ID:
    print("[AGENT] WARNING: RUN_ID is not set")
if not SANDBOX_PROXY_URL:
    print("[AGENT] WARNING: SANDBOX_PROXY_URL is not set")

def inference(model, temperature, messages):
    """Official-style inference helper"""
    try:
        payload = {"run_id": RUN_ID, "model": model, "temperature": temperature, "messages": messages}
        print(f"[AGENT] Sending inference request for model {model}")
        import requests
        response = requests.post(
            f"{SANDBOX_PROXY_URL}/api/inference",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload)
        )
        if response.status_code == 200:
            result = response.text.strip('"')
            print(f"[AGENT] Inference success: {len(result)} characters")
            return result
        else:
            print(f"[AGENT] Inference failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"[AGENT] Inference error: {e}")
        return None

def get_research_context(problem: str) -> str:
    """Pull from our research_bot.log if available"""
    try:
        log_path = Path.home() / "bittensor-claw" / "research_bot.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-80:]
            return "".join(lines[-50:])
    except Exception:
        pass
    return "No recent research data. Focus on clean, minimal fix."

def agent_main(input: Dict[str, Any]) -> str:
    """Official-compliant main entrypoint"""
    print("[AGENT] Entered agent_main()")
    
    problem = input.get("problem_statement", "")
    if not problem:
        print("[AGENT] No problem_statement provided")
        return ""

    # Get research context
    context = get_research_context(problem)
    print(f"[AGENT] Research context loaded: {len(context)} characters")

    # Simple but smart prompt for diff generation
    messages = [
        {"role": "system", "content": "You are an elite coding agent. Solve the problem with a minimal, correct git diff. Use research context for robustness."},
        {"role": "user", "content": f"Problem: {problem}\n\nResearch Context:\n{context}\n\nGenerate ONLY a valid git diff to solve this."}
    ]

    diff = inference("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 0.3, messages)
    if not diff:
        print("[AGENT] Failed to generate diff")
        return ""

    # Write the diff to the expected location (official requirement)
    solution_path = Path("/sandbox/solution.diff")
    solution_path.parent.mkdir(parents=True, exist_ok=True)
    with open(solution_path, "w", encoding="utf-8") as f:
        f.write(diff)

    print(f"[AGENT] Wrote diff to /sandbox/solution.diff ({len(diff)} characters)")
    print("[AGENT] Exiting agent_main()")

    return diff

# For local testing
if __name__ == "__main__":
    print("Ridges SN62 Agent loaded successfully.")
    print("LOCAL TEST MODE: agent_main() is ready.")
    print("In the real SN62 sandbox, it will use /sandbox/solution.diff automatically.")
