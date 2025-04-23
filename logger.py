# logger.py
import os
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "feature_test_log.txt")

def log_action(action, status, message=""):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {action} - {status}: {message}\n"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
