"""Environment-based application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
# llama-3.3-70b-versatile has been decommissioned by Groq; check the models
# endpoint if this default ever starts returning model_not_found.
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
DB_PATH = Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "credit_risk.db")))
RISK_THRESHOLD_LOW = float(os.getenv("RISK_THRESHOLD_LOW", "0.3"))
RISK_THRESHOLD_HIGH = float(os.getenv("RISK_THRESHOLD_HIGH", "0.6"))
