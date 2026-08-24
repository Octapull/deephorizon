"""Check MLflow SQLite database contents."""
import sqlite3
from pathlib import Path

db_path = Path("mlflow.db")
if not db_path.exists():
    print(f"mlflow.db not found at {db_path.absolute()}")
    exit(1)

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Tablolar
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("=== TABLOLAR ===")
for t in tables:
    print(f"  {t[0]}")

# Experiments
print()
print("=== EXPERIMENTS ===")
cursor.execute("SELECT experiment_id, name, artifact_location FROM experiments")
for row in cursor.fetchall():
    print(f"  ID: {row[0]}, Name: {row[1]}, Location: {row[2]}")

# Runs
print()
print("=== RUNS ===")
cursor.execute(
    "SELECT run_uuid, experiment_id, status, start_time, end_time FROM runs LIMIT 20"
)
runs = cursor.fetchall()
for row in runs:
    print(f"  UUID: {row[0][:8]}..., Exp: {row[1]}, Status: {row[2]}")

print(f"\nToplam run: {len(runs)}")

# Metrics (son 20)
print()
print("=== METRICS (son 20) ===")
cursor.execute(
    """
    SELECT run_uuid, key, value, timestamp
    FROM metrics
    ORDER BY timestamp DESC
    LIMIT 20
"""
)
for row in cursor.fetchall():
    print(f"  Run: {row[0][:8]}..., Key: {row[1]}, Value: {row[2]:.4f}")

# Params (son 20)
print()
print("=== PARAMS (son 20) ===")
cursor.execute(
    """
    SELECT run_uuid, key, value
    FROM params
    LIMIT 20
"""
)
for row in cursor.fetchall():
    print(f"  Run: {row[0][:8]}..., Key: {row[1]}, Value: {row[2]}")

conn.close()
