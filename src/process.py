import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("../data/raw")
OUT_DIR = Path("../data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS_PATH = RAW_DIR / "telemetry_logs.jsonl"
EMP_PATH = RAW_DIR / "employees.csv"

def parse_cloudwatch_export(jsonl_path: Path) -> pd.DataFrame:
    """
    telemetry_logs.jsonl lines look like CloudWatch Logs export:
    each line has 'logEvents': [{id,timestamp,message}, ...]
    We explode logEvents to a flat table.
    """
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            owner = obj.get("owner")
            log_group = obj.get("logGroup")
            log_stream = obj.get("logStream")
            y, m, d = obj.get("year"), obj.get("month"), obj.get("day")

            events = obj.get("logEvents", []) or []
            for ev in events:
                rows.append({
                    "owner": owner,
                    "log_group": log_group,
                    "log_stream": log_stream,
                    "year": y,
                    "month": m,
                    "day": d,
                    "event_id": ev.get("id"),
                    "timestamp_ms": ev.get("timestamp"),
                    "message": ev.get("message"),
                })

    df = pd.DataFrame(rows)
    return df

def extract_fields_from_message(df: pd.DataFrame) -> pd.DataFrame:
    """
    message is often JSON string; if not, keep raw.
    We try to parse JSON and extract common analytics fields like:
    event_type, tokens, session_id, request_type, etc.
    """
    # Try parse message JSON
    parsed = []
    for msg in df["message"].fillna("").astype(str):
        msg = msg.strip()
        if msg.startswith("{") and msg.endswith("}"):
            try:
                parsed.append(json.loads(msg))
            except Exception:
                parsed.append({})
        else:
            parsed.append({})

    df_msg = pd.json_normalize(parsed)
    # Merge back
    out = pd.concat([df.reset_index(drop=True), df_msg.reset_index(drop=True)], axis=1)

    # Normalize likely columns
    rename_map = {
        "email": "email",
        "user.email": "email",
        "userEmail": "email",
        "user_email": "email",
        "userId": "user_id",
        "sessionId": "session_id",
        "session_id": "session_id",
        "eventType": "event_type",
        "event_type": "event_type",
        "token_count": "tokens",
        "tokens": "tokens",
        "promptTokens": "prompt_tokens",
        "completionTokens": "completion_tokens",
        "projectType": "project_type",
        "project_type": "project_type",
    }
    for k, v in rename_map.items():
        if k in out.columns and v not in out.columns:
            out = out.rename(columns={k: v})

    # If we don't have email in message, derive it from 'owner' if it looks like an email
    if "email" not in out.columns:
        out["email"] = None
    out["email"] = out["email"].fillna(
        out["owner"].where(out["owner"].astype(str).str.contains("@"), None)
    )

    # Timestamps
    out["ts"] = pd.to_datetime(out["timestamp_ms"], unit="ms", errors="coerce", utc=True)

    # Tokens fallback: try compute from prompt+completion if present
    if "tokens" in out.columns:
        out["tokens"] = pd.to_numeric(out["tokens"], errors="coerce").fillna(0)
    else:
        out["tokens"] = 0

    if "prompt_tokens" in out.columns or "completion_tokens" in out.columns:
        pt = pd.to_numeric(out.get("prompt_tokens", 0), errors="coerce").fillna(0)
        ct = pd.to_numeric(out.get("completion_tokens", 0), errors="coerce").fillna(0)
        out["tokens"] = out["tokens"].where(out["tokens"] != 0, pt + ct)

    out["tokens"] = out["tokens"].fillna(0).astype(int)

    # If no event_type, create something from messageType/log_group/etc
    if "event_type" not in out.columns:
        out["event_type"] = out.get("messageType", "unknown")

    # Time features
    out["date"] = out["ts"].dt.date
    out["hour"] = out["ts"].dt.hour

    return out

def main():
    df_emp = pd.read_csv(EMP_PATH)
    df_raw = parse_cloudwatch_export(LOGS_PATH)
    df = extract_fields_from_message(df_raw)
    # FIX: email is stored in attributes.user.email
    if "email" not in df.columns or df["email"].isna().all():
        if "attributes.user.email" in df.columns:
            df["email"] = df["attributes.user.email"]

    if "practice" in df.columns:
        df.groupby("practice", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False)\
            .to_csv(OUT_DIR / "tokens_by_practice.csv", index=False)

    if "location" in df.columns:
        df.groupby("location", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False)\
            .to_csv(OUT_DIR / "tokens_by_location.csv", index=False)

    # Usage over time
    df.groupby("hour", dropna=False)["tokens"].sum().reset_index().sort_values("hour")\
        .to_csv(OUT_DIR / "tokens_by_hour.csv", index=False)

    df.groupby("date", dropna=False)["tokens"].sum().reset_index().sort_values("date")\
        .to_csv(OUT_DIR / "tokens_by_day.csv", index=False)

    # Sessions summary if exists, else do by log_stream
    if "session_id" in df.columns:
        key = "session_id"
    else:
        key = "log_stream"

    sess = df.groupby(key, dropna=False).agg(
        events=("message", "count"),
        tokens=("tokens", "sum"),
        users=("email", "nunique"),
    ).reset_index()
    sess["tokens_per_event"] = (sess["tokens"] / sess["events"]).round(2)
    sess = sess.sort_values("tokens", ascending=False)
    sess.to_csv(OUT_DIR / "sessions_summary.csv", index=False)

    print("Saved processed files to:", OUT_DIR.resolve())
    print("Processed events shape:", df.shape)
    print("Non-null emails:", int(df["email"].notna().sum()))
    print("Example columns:", list(df.columns)[:25])

if __name__ == "__main__":
    main()