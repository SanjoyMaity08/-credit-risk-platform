"""
Execute SQL against the SQLite database and return results.
"""
import os
import sqlite3
import pandas as pd
from src.utils.config import DB_PATH
from src.utils.logger import get_logger

logger = get_logger(__name__)

BLOCKED_KEYWORDS = {"drop", "delete", "insert", "update", "alter", "create", "truncate"}


def validate_sql(sql: str) -> bool:
    """Basic allowlist / blocklist validation."""
    if sql.strip().upper() == "UNSUPPORTED":
        return False
    tokens = set(sql.lower().split())
    if tokens & BLOCKED_KEYWORDS:
        logger.warning(f"Blocked SQL keywords detected: {tokens & BLOCKED_KEYWORDS}")
        return False
    return True


def run_query(sql: str) -> tuple:
    """
    Execute a SQL query against the SQLite DB.
    Returns (DataFrame | None, error_message | None).
    """
    if not validate_sql(sql):
        return None, "Query blocked: contains disallowed keywords or is marked UNSUPPORTED."

    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df, None
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        return None, str(e)


def load_data_to_db(df: pd.DataFrame, table_name: str = "applications"):
    """Load a dataframe into SQLite for querying."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    conn.close()
    logger.info(f"Loaded {len(df)} rows into table '{table_name}' at {DB_PATH}")
