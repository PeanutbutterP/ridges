import json
import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Sandbox environment variables
SANDBOX_PROXY_URL = os.getenv("SANDBOX_PROXY_URL")
EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID")
AGENT_TIMEOUT = int(os.getenv("AGENT_TIMEOUT", 1500))

# Safe directory change
try:
    if Path("/sandbox/repo").exists():
        os.chdir("/sandbox/repo")
        print("[SANDBOX MODE] Working in /sandbox/repo")
    else:
        print("[LOCAL TEST MODE] Running outside sandbox")
except Exception:
    print("[LOCAL TEST MODE] Running outside sandbox")

# Globals
proxy_cost = 0
MAX_COST = 100
start_time = time.time()
failure_patterns = []
success_patterns = []

def track_cost(model: str):
    global proxy_cost
    cost = 10 if "Qwen" in model else 2
    proxy_cost += cost
    if proxy_cost > MAX_COST:
        raise Exception(f"Cost limit exceeded: {proxy_cost}")

def switch_model():
    global proxy_cost
    elapsed = time.time() - start_time
    if proxy_cost > MAX_COST * 0.5 or elapsed > AGENT_TIMEOUT * 0.6:
        return "gpt-4o-mini"
    return "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"

def call_inference(
    model: str,
    temperature: float,
    messages: List[Dict[str, str]],
    tools: Optional[List[Dict[str, Any]]] = None,
    max_retries: int = 3
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    model = switch_model() if model == "auto" else model
    track_cost(model)
    if not SANDBOX_PROXY_URL or not EVALUATION_RUN_ID:
        return None, []

    payload = {
        "evaluation_run_id": EVALUATION_RUN_ID,
        "model": model,
        "temperature": temperature,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_mode"] = "auto"

    for attempt in range(max_retries):
        try:
            import requests
            resp = requests.post(f"{SANDBOX_PROXY_URL}/api/inference", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            return data.get("content"), data.get("tool_calls", [])
        except Exception as e:
            print(f"[RETRY {attempt+1}] Inference failed: {e}")
            time.sleep(2 ** attempt)
    return None, []

def get_research_context(problem: str) -> str:
    try:
        log_path = Path.home() / "bittensor-claw" / "research_bot.log"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[-100:]
            return "".join(lines[-60:])
    except Exception:
        pass
    return "No recent research data available."

def analyze_repo() -> str:
    try:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=10)
        return f"Repo status:\n{status.stdout.strip() or 'Clean working tree'}"
    except Exception as e:
        return f"Repo analysis failed: {e}"

def embed_repo_files() -> Dict[str, List[float]]:
    embeddings = {}
    def embed_file(f):
        try:
            with open(f, "r", encoding="utf-8") as file:
                content = file.read()
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                for i, chunk in enumerate(chunks):
                    emb = call_embedding(chunk)
                    if emb:
                        embeddings[f"{f}_chunk_{i}"] = emb
        except:
            pass

    files = [f for f in os.listdir(".") if f.endswith((".py", ".md", ".txt"))][:20]
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(embed_file, files)
    return embeddings

def call_embedding(text: str) -> Optional[List[float]]:
    track_cost("embedding")
    if not SANDBOX_PROXY_URL or not EVALUATION_RUN_ID:
        return None
    payload = {"evaluation_run_id": EVALUATION_RUN_ID, "model": "text-embedding-3-small", "input": text}
    try:
        import requests
        resp = requests.post(f"{SANDBOX_PROXY_URL}/api/embedding", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json().get("embedding")
    except:
        return None

def rag_search(query: str, embeddings: Dict[str, List[float]]) -> str:
    query_emb = call_embedding(query)
    if not query_emb or not embeddings:
        return ""
    def cos_sim(a, b):
        return sum(x * y for x, y in zip(a, b)) / (sum(x**2 for x in a)**0.5 * sum(y**2 for y in b)**0.5 + 1e-8)
    top = sorted(embeddings, key=lambda k: cos_sim(query_emb, embeddings[k]), reverse=True)[:3]
    return f"Top matches: {', '.join(top)}"

def run_tests() -> Tuple[bool, str]:
    try:
        result = subprocess.run(["python", "-m", "pytest", "--tb=short", "-q", "-x"], capture_output=True, text=True, timeout=120)
        passed = result.returncode == 0
        return passed, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)

def generate_and_test_fix(problem: str, context: str, repo_analysis: str, embeddings: Dict[str, List[float]]) -> str:
    diff = ""
    for iteration in range(4):
        if time.time() - start_time > AGENT_TIMEOUT * 0.75:
            break

        rag = rag_search(problem, embeddings)
        messages = [
            {"role": "system", "content": "Elite SWE-Bench fixer. Use RAG, tools, avoid past failure patterns."},
            {"role": "user", "content": f"Problem: {problem}\nContext: {context}\nRepo: {repo_analysis}\nRAG: {rag}\nIteration {iteration+1}: Generate diff."}
        ]
        content, _ = call_inference("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 0.25, messages)
        diff = content or ""

        if diff:
            try:
                subprocess.run(["git", "apply", "--ignore-whitespace"], input=diff, text=True, timeout=20)
                passed, output = run_tests()
                if passed:
                    success_patterns.append(problem[:50])
                    return diff
                failure_patterns.append(output[:100])
                subprocess.run(["git", "reset", "--hard"], timeout=10)
            except:
                pass
    return diff

def agent_main(input: Dict[str, Any]) -> str:
    global start_time
    problem = input.get("problem_statement", "")
    if not problem:
        return ""

    context = get_research_context(problem)
    repo_analysis = analyze_repo()
    embeddings = embed_repo_files()

    diff = generate_and_test_fix(problem, context, repo_analysis, embeddings)

    return diff.strip()

# Local test
if __name__ == "__main__":
    print("Ridges SN62 Agent loaded successfully.")
    print("LOCAL TEST MODE: agent_main() is ready.")
    print("In the real SN62 sandbox, it will use /sandbox/repo automatically.")
