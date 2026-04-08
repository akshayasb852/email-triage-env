---
title: Email Triage Environment
emoji: 📧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 📧 Email Triage Environment

A real-world OpenEnv environment where an AI agent learns to triage emails into **urgent**, **normal**, or **spam** categories with priority assignment.

## 🌍 Real-World Utility

Email triage is a task every professional performs daily. This environment trains agents to:
- Classify emails by urgency
- Assign priority levels (1-5)
- Generate concise summaries

## 🎯 Tasks

| Task | Difficulty | Emails | Description |
|------|-----------|--------|-------------|
| easy | ⭐ Easy | 6 | Classify urgent vs spam only |
| medium | ⭐⭐ Medium | 9 | Classify urgent, normal, spam |
| hard | ⭐⭐⭐ Hard | 15 | All emails with correct priority |

## 📊 Observation Space

```json
{
  "email_id": "string",
  "subject": "string", 
  "sender": "string",
  "body": "string",
  "timestamp": "string",
  "task_name": "string",
  "step": "integer",
  "max_steps": "integer",
  "score": "float"
}
```

## ⚡ Action Space

```json
{
  "label": "urgent | normal | spam",
  "priority": "1-5 (1=highest)",
  "summary": "one-line email summary"
}
```

## 🏆 Reward Function

| Component | Weight | Description |
|-----------|--------|-------------|
| Label correct | 0.5 | Right category |
| Priority correct | 0.3 | Exact priority match |
| Priority close | 0.15 | Off by 1 |
| Summary quality | 0.2 | Meaningful summary |

## 🚀 Setup

```bash
pip install -r requirements.txt
python app.py
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Take action |
| `/state` | GET | Get current state |
| `/tasks` | GET | List all tasks |
| `/score` | GET | Get current score |

## 📈 Baseline Scores

| Task | Score | Steps |
|------|-------|-------|
| easy | 0.167 | 6 |
| medium | 0.394 | 9 |
| hard | 0.397 | 15 |

## 🔧 Environment Variables

```
HF_TOKEN=your_huggingface_token
OPENAI_API_KEY=your_api_key
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama3-8b-8192
```

## 🐳 Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```