import os
import json
import requests
from openai import OpenAI

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "dummy"))
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "https://akshayasb-email-triage-openenv.hf.space")

TASKS = ["easy", "medium", "hard"]
BENCHMARK = "email-triage"
MAX_STEPS = 20
SUCCESS_SCORE_THRESHOLD = 0.5

print(f"[DEBUG] API_BASE_URL={API_BASE_URL}", flush=True)
print(f"[DEBUG] MODEL_NAME={MODEL_NAME}", flush=True)
print(f"[DEBUG] ENV_BASE_URL={ENV_BASE_URL}", flush=True)

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception as e:
    print(f"[DEBUG] Client error: {e}", flush=True)
    client = None

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

SYSTEM_PROMPT = """You are an email triage assistant. Classify emails as urgent, normal, or spam.
Respond ONLY with valid JSON: {"label": "urgent|normal|spam", "priority": 1-5, "summary": "brief summary"}"""

def get_action(obs):
    prompt = f"Subject: {obs.get('subject','')}\nFrom: {obs.get('sender','')}\nBody: {obs.get('body','')}"
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0,
    )
    text = resp.choices[0].message.content.strip()
    if "```" in text:
        text = text.split("```")[1].replace("json","").strip()
    action = json.loads(text)
    if action.get("label") not in ["urgent","normal","spam"]:
        action["label"] = "normal"
    if not isinstance(action.get("priority"), int):
        action["priority"] = 3
    if not action.get("summary"):
        action["summary"] = "email summary"
    return action

def run_task(task_name):
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    log_start(task_name, BENCHMARK, MODEL_NAME)
    try:
        r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_name": task_name}, timeout=30)
        obs = r.json().get("observation", {})
        for step in range(1, MAX_STEPS + 1):
            action = get_action(obs)
            action_str = f"label={action['label']},priority={action['priority']}"
            sr = requests.post(f"{ENV_BASE_URL}/step", json={"task_name": task_name, "label": action["label"], "priority": action["priority"], "summary": action["summary"]}, timeout=30)
            step_result = sr.json()
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            error = step_result.get("info", {}).get("error", None)
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, error)
            if done:
                break
            obs = step_result.get("observation", obs)
        sr = requests.get(f"{ENV_BASE_URL}/score?task_name={task_name}", timeout=30)
        score = sr.json().get("score", 0.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] Task error: {e}", flush=True)
        score = 0.0
        success = False
    finally:
        log_end(success, steps_taken, score, rewards)
    return {"task": task_name, "score": score, "success": success}

def main():
    print("[DEBUG] Starting inference", flush=True)
    results = []
    for task in TASKS:
        print(f"[DEBUG] Running {task}", flush=True)
        results.append(run_task(task))
    print("[DEBUG] ===== RESULTS =====", flush=True)
    for r in results:
        print(f"[DEBUG] {r['task']}: score={r['score']:.3f} success={r['success']}", flush=True)

if __name__ == "__main__":
    main()