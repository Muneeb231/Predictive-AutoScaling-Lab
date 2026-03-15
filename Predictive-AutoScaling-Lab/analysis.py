import pandas as pd

def summarize(path):
    df = pd.read_csv(path)
    return {
        "avg_instances": df["instances"].mean(),
        "total_cost": df["cost"].sum(),
        "sla_violations": df["sla_violation"].sum(),
        "scale_outs": df["scaled_up"].sum(),
        "scale_ins": df["scaled_down"].sum(),
        "oscillations": (df["scaled_up"] + df["scaled_down"]).sum()
    }

results = {
    "reactive": summarize("reactive_results.csv"),
    "shifted": summarize("shifted_results.csv"),
    "moving_avg": summarize("moving_avg_results.csv"),
    # Add ARIMA later
}

df_summary = pd.DataFrame(results).T
print(df_summary)
df_summary.to_csv("summary_table.csv")
