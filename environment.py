import random
from typing import Optional
from pydantic import BaseModel


# ─── Pydantic Models ───────────────────────────────────────────────────────────

class EmailObservation(BaseModel):
    email_id: str
    subject: str
    sender: str
    body: str
    timestamp: str
    task_name: str
    step: int
    max_steps: int
    score: float


class EmailAction(BaseModel):
    label: str          # "urgent", "normal", or "spam"
    priority: int       # 1 (highest) to 5 (lowest)
    summary: str        # one-line summary of the email


class EmailReward(BaseModel):
    reward: float
    label_correct: bool
    priority_correct: bool
    summary_quality: float
    done: bool


# ─── Email Dataset ─────────────────────────────────────────────────────────────

EMAILS = [
    # URGENT emails
    {
        "email_id": "e001",
        "subject": "URGENT: Server down in production",
        "sender": "alerts@company.com",
        "body": "Our production server crashed 10 minutes ago. All services are down. Immediate action required.",
        "timestamp": "2024-01-15 09:00:00",
        "true_label": "urgent",
        "true_priority": 1,
    },
    {
        "email_id": "e002",
        "subject": "Critical security breach detected",
        "sender": "security@company.com",
        "body": "We have detected unauthorized access to customer data. Please respond immediately.",
        "timestamp": "2024-01-15 09:05:00",
        "true_label": "urgent",
        "true_priority": 1,
    },
    {
        "email_id": "e003",
        "subject": "CEO meeting in 30 minutes - mandatory attendance",
        "sender": "ceo@company.com",
        "body": "All department heads must attend the emergency meeting in 30 minutes. Crisis situation.",
        "timestamp": "2024-01-15 09:10:00",
        "true_label": "urgent",
        "true_priority": 2,
    },
    {
        "email_id": "e004",
        "subject": "Database backup failed - data at risk",
        "sender": "dba@company.com",
        "body": "Last night backup failed. 3 days of data unprotected. Need immediate fix.",
        "timestamp": "2024-01-15 09:15:00",
        "true_label": "urgent",
        "true_priority": 1,
    },
    {
        "email_id": "e005",
        "subject": "Payment processing down - revenue impact",
        "sender": "ops@company.com",
        "body": "Payment gateway is not responding. Customers cannot complete purchases. Fix ASAP.",
        "timestamp": "2024-01-15 09:20:00",
        "true_label": "urgent",
        "true_priority": 1,
    },
    # NORMAL emails
    {
        "email_id": "e006",
        "subject": "Weekly team meeting agenda",
        "sender": "manager@company.com",
        "body": "Please find attached the agenda for our weekly team meeting on Friday at 2pm.",
        "timestamp": "2024-01-15 10:00:00",
        "true_label": "normal",
        "true_priority": 3,
    },
    {
        "email_id": "e007",
        "subject": "Project update - Q1 milestones",
        "sender": "pm@company.com",
        "body": "Hi team, attaching the Q1 milestone tracker. Please review and update your tasks by EOD.",
        "timestamp": "2024-01-15 10:30:00",
        "true_label": "normal",
        "true_priority": 3,
    },
    {
        "email_id": "e008",
        "subject": "New office supplies order",
        "sender": "admin@company.com",
        "body": "We are placing an office supplies order. Please submit your requests by Thursday.",
        "timestamp": "2024-01-15 11:00:00",
        "true_label": "normal",
        "true_priority": 4,
    },
    {
        "email_id": "e009",
        "subject": "HR: Updated leave policy 2024",
        "sender": "hr@company.com",
        "body": "Please read the attached updated leave policy document effective from February 2024.",
        "timestamp": "2024-01-15 11:30:00",
        "true_label": "normal",
        "true_priority": 3,
    },
    {
        "email_id": "e010",
        "subject": "Code review request - feature branch",
        "sender": "developer@company.com",
        "body": "Could you review my pull request for the new login feature? PR link: github.com/pr/123",
        "timestamp": "2024-01-15 12:00:00",
        "true_label": "normal",
        "true_priority": 3,
    },
    # SPAM emails
    {
        "email_id": "e011",
        "subject": "You WON $1,000,000! Claim NOW!!!",
        "sender": "noreply@scam123.com",
        "body": "Congratulations! You have been selected as our lucky winner. Click here to claim your prize!",
        "timestamp": "2024-01-15 08:00:00",
        "true_label": "spam",
        "true_priority": 5,
    },
    {
        "email_id": "e012",
        "subject": "FREE iPhone 15 - Limited offer!!!",
        "sender": "offers@fakedeals.net",
        "body": "Get your FREE iPhone 15 today! Only 100 left. Click the link below immediately!",
        "timestamp": "2024-01-15 08:05:00",
        "true_label": "spam",
        "true_priority": 5,
    },
    {
        "email_id": "e013",
        "subject": "Cheap medications - no prescription needed",
        "sender": "pharma@illegal-drugs.com",
        "body": "Buy any medication without prescription. Huge discounts. Worldwide shipping.",
        "timestamp": "2024-01-15 08:10:00",
        "true_label": "spam",
        "true_priority": 5,
    },
    {
        "email_id": "e014",
        "subject": "Your account will be suspended - verify NOW",
        "sender": "security@fake-bank.xyz",
        "body": "Your bank account will be suspended in 24 hours. Click here to verify your identity.",
        "timestamp": "2024-01-15 08:15:00",
        "true_label": "spam",
        "true_priority": 5,
    },
    {
        "email_id": "e015",
        "subject": "Make $5000 per day working from home!!!",
        "sender": "jobs@get-rich-quick.biz",
        "body": "Earn $5000 every day from home! No experience needed. Join thousands of happy members!",
        "timestamp": "2024-01-15 08:20:00",
        "true_label": "spam",
        "true_priority": 5,
    },
]


# ─── Environment ───────────────────────────────────────────────────────────────

class EmailTriageEnv:
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.emails = []
        self.current_index = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.max_steps = 0
        self.done = False
        self._setup_task()

    def _setup_task(self):
        if self.task_name == "easy":
            # Easy: only urgent vs spam (very different)
            self.emails = [e for e in EMAILS if e["true_label"] in ["urgent", "spam"]][:6]
            self.max_steps = len(self.emails)
        elif self.task_name == "medium":
            # Medium: all 3 labels, balanced
            self.emails = [e for e in EMAILS if e["true_label"] == "urgent"][:3] + \
                          [e for e in EMAILS if e["true_label"] == "normal"][:3] + \
                          [e for e in EMAILS if e["true_label"] == "spam"][:3]
            self.max_steps = len(self.emails)
        elif self.task_name == "hard":
            # Hard: all emails, must also get priority right
            self.emails = EMAILS.copy()
            self.max_steps = len(self.emails)
        random.shuffle(self.emails)

    def reset(self) -> EmailObservation:
        self._setup_task()
        self.current_index = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self) -> EmailObservation:
        if self.current_index >= len(self.emails):
            email = self.emails[-1]
        else:
            email = self.emails[self.current_index]
        return EmailObservation(
            email_id=email["email_id"],
            subject=email["subject"],
            sender=email["sender"],
            body=email["body"],
            timestamp=email["timestamp"],
            task_name=self.task_name,
            step=self.step_count,
            max_steps=self.max_steps,
            score=self.total_reward,
        )

    def step(self, action: EmailAction):
        if self.done:
            obs = self._get_observation()
            return obs, 0.0, True, {"error": "Episode already done"}

        email = self.emails[self.current_index]
        reward = self._compute_reward(action, email)

        self.total_reward += reward
        self.step_count += 1
        self.current_index += 1

        if self.current_index >= len(self.emails):
            self.done = True

        obs = self._get_observation()
        info = {
            "true_label": email["true_label"],
            "true_priority": email["true_priority"],
            "label_correct": action.label == email["true_label"],
            "reward": reward,
        }
        return obs, reward, self.done, info

    def _compute_reward(self, action: EmailAction, email: dict) -> float:
        reward = 0.0

        # Label correctness (0.5 weight)
        if action.label == email["true_label"]:
            reward += 0.5

        # Priority correctness (0.3 weight)
        priority_diff = abs(action.priority - email["true_priority"])
        if priority_diff == 0:
            reward += 0.3
        elif priority_diff == 1:
            reward += 0.15
        elif priority_diff == 2:
            reward += 0.05

        # Summary quality (0.2 weight) - check if summary is meaningful
        if action.summary and len(action.summary) > 10:
            reward += 0.1
        if action.summary and len(action.summary) > 30:
            reward += 0.1

        return round(reward, 2)

    def state(self) -> dict:
        return {
            "task_name": self.task_name,
            "current_index": self.current_index,
            "total_emails": len(self.emails),
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "done": self.done,
        }

    def get_final_score(self) -> float:
        max_possible = self.max_steps * 1.0
        if max_possible == 0:
            return 0.0
        return round(min(self.total_reward / max_possible, 1.0), 3)