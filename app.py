from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import EmailTriageEnv, EmailAction

app = FastAPI(
    title="Email Triage Environment",
    description="An OpenEnv-compatible environment for email triage tasks",
    version="1.0.0",
)

# Global environment instances for each task
envs = {
    "easy": EmailTriageEnv(task_name="easy"),
    "medium": EmailTriageEnv(task_name="medium"),
    "hard": EmailTriageEnv(task_name="hard"),
}

current_task = "easy"


class ResetRequest(BaseModel):
    task_name: Optional[str] = "easy"


class StepRequest(BaseModel):
    task_name: Optional[str] = "easy"
    label: str
    priority: int
    summary: str


@app.get("/")
def root():
    return {
        "name": "Email Triage Environment",
        "version": "1.0.0",
        "tasks": ["easy", "medium", "hard"],
        "description": "AI agent learns to triage emails into urgent/normal/spam categories",
    }


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    task_name = request.task_name or "easy"
    if task_name not in envs:
        task_name = "easy"
    obs = envs[task_name].reset()
    return {
        "observation": obs.dict(),
        "task_name": task_name,
        "info": "Environment reset successfully",
    }


@app.post("/step")
def step(request: StepRequest):
    task_name = request.task_name or "easy"
    if task_name not in envs:
        task_name = "easy"

    action = EmailAction(
        label=request.label,
        priority=request.priority,
        summary=request.summary,
    )

    obs, reward, done, info = envs[task_name].step(action)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state(task_name: str = "easy"):
    if task_name not in envs:
        task_name = "easy"
    return envs[task_name].state()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "easy",
                "description": "Classify emails as urgent or spam only",
                "difficulty": "easy",
                "num_emails": 6,
            },
            {
                "name": "medium",
                "description": "Classify emails as urgent, normal, or spam",
                "difficulty": "medium",
                "num_emails": 9,
            },
            {
                "name": "hard",
                "description": "Classify all emails with correct priority levels",
                "difficulty": "hard",
                "num_emails": 15,
            },
        ]
    }


@app.get("/score")
def get_score(task_name: str = "easy"):
    if task_name not in envs:
        task_name = "easy"
    return {
        "task_name": task_name,
        "score": envs[task_name].get_final_score(),
        "total_reward": envs[task_name].total_reward,
        "steps": envs[task_name].step_count,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)