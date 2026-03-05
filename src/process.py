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
                rows.append(
                    {
                        "owner": owner,
                        "log_group": log_group,
                        "log_stream": log_stream,
                        "year": y,
                        "month": m,
                        "day": d,
                        "event_id": ev.get("id"),
                        "timestamp_ms": ev.get("timestamp"),
                        "message": ev.get("message"),
                    }
                )

    return pd.DataFrame(rows)


def extract_fields_from_message(df: pd.DataFrame) -> pd.DataFrame:
    """
    message is sometimes a JSON string. We try to parse it and flatten.
    Then we create normalized columns used for analytics:
    - ts (timestamp)
    - tokens (fallback to attributes.prompt_length if tokens missing/zero)
    - email (from attributes.user.email)
    - session_id (from attributes.session.id)
    """
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
    out = pd.concat([df.reset_index(drop=True), df_msg.reset_index(drop=True)], axis=1)

    # Rename a few common variants (if they exist)
    rename_map = {
        "sessionId": "session_id",
        "eventType": "event_type",
        "token_count": "tokens",
        "promptTokens": "prompt_tokens",
        "completionTokens": "completion_tokens",
        "projectType": "project_type",
        "userEmail": "email",
        "user.email": "email",
    }
    for k, v in rename_map.items():
        if k in out.columns and v not in out.columns:
            out = out.rename(columns={k: v})

    # Timestamp
    out["ts"] = pd.to_datetime(out["timestamp_ms"], unit="ms", errors="coerce", utc=True)

    # Email (real one in your dataset)
    if "attributes.user.email" in out.columns:
        out["email"] = out["attributes.user.email"]
    elif "email" not in out.columns:
        out["email"] = None

    # Session id (real one in your dataset)
    if "attributes.session.id" in out.columns:
        out["session_id"] = out["attributes.session.id"]
    elif "session_id" not in out.columns:
        out["session_id"] = None

    # Tokens (try direct, then fallback to prompt_length)
    if "tokens" in out.columns:
        out["tokens"] = pd.to_numeric(out["tokens"], errors="coerce").fillna(0)
    else:
        out["tokens"] = 0

    # If prompt/completion tokens exist, use them when tokens are 0
    if ("prompt_tokens" in out.columns) or ("completion_tokens" in out.columns):
        pt = pd.to_numeric(out.get("prompt_tokens", 0), errors="coerce").fillna(0)
        ct = pd.to_numeric(out.get("completion_tokens", 0), errors="coerce").fillna(0)
        # fill only where tokens are 0
        out["tokens"] = out["tokens"].where(out["tokens"] != 0, pt + ct)

    # Final fallback: prompt_length as proxy if tokens are still all 0
    if out["tokens"].sum() == 0 and "attributes.prompt_length" in out.columns:
        out["tokens"] = (
            pd.to_numeric(out["attributes.prompt_length"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        out["tokens"] = out["tokens"].fillna(0).astype(int)

    # Event type fallback
    if "attributes.event.name" in out.columns and "event_type" not in out.columns:
        out["event_type"] = out["attributes.event.name"]
    elif "event_type" not in out.columns:
        out["event_type"] = "unknown"

    # Time features
    out["date"] = out["ts"].dt.date
    out["hour"] = out["ts"].dt.hour

    return out


def main():
    df_emp = pd.read_csv(EMP_PATH)

    df_raw = parse_cloudwatch_export(LOGS_PATH)
    df = extract_fields_from_message(df_raw)

    # Merge with employees by email (employees columns: email, full_name, practice, level, location)
    df = df.merge(df_emp, on="email", how="left")

    # Save processed events
    df.to_csv(OUT_DIR / "events_processed.csv", index=False)

    # Aggregates (these exist now for dashboard)
    df.groupby("level", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False).to_csv(
        OUT_DIR / "tokens_by_level.csv", index=False
    )
    df.groupby("practice", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False).to_csv(
        OUT_DIR / "tokens_by_practice.csv", index=False
    )
    df.groupby("location", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False).to_csv(
        OUT_DIR / "tokens_by_location.csv", index=False
    )

    df.groupby("hour", dropna=False)["tokens"].sum().reset_index().sort_values("hour").to_csv(
        OUT_DIR / "tokens_by_hour.csv", index=False
    )
    df.groupby("date", dropna=False)["tokens"].sum().reset_index().sort_values("date").to_csv(
        OUT_DIR / "tokens_by_day.csv", index=False
    )

    # Sessions summary
    sess = df.groupby("session_id", dropna=False).agg(
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
    print("Total tokens:", int(df["tokens"].sum()))
    print("Unique sessions:", int(df["session_id"].nunique()))


if __name__ == "__main__":
    main()