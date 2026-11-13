"""Central configuration — reads from environment variables."""
import os

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models")
DB_PATH = os.environ.get("DB_PATH", "/app/data/credit_risk.db")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 8501))
