import pandas as pd
import json

logs = []

with open("../data/raw/telemetry_logs.jsonl", "r") as f:
    for line in f:
        logs.append(json.loads(line))

df_logs = pd.DataFrame(logs)

df_employees = pd.read_csv("../data/raw/employees.csv")

print("Logs shape:", df_logs.shape)
print("Employees shape:", df_employees.shape)

print(df_logs.head())
print(df_employees.head())