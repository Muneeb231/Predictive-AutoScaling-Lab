import polars as pl
import matplotlib.pyplot as plt

# Load cleaned data
df = pl.read_csv("app_memory_percentiles_clean.csv")

# Convert to pandas for plotting
pdf = df.to_pandas()

# 1. Histogram of AverageAllocatedMb
plt.figure()
plt.hist(pdf["AverageAllocatedMb"].dropna(), bins=40)
plt.title("Distribution of Average Allocated Memory (MB)")
plt.xlabel("MB")
plt.ylabel("Frequency")
plt.savefig("plot_memory_hist.png")


# 2. Percentile comparison plot
percentile_cols = [
    "AverageAllocatedMb_pct1",
    "AverageAllocatedMb_pct5",
    "AverageAllocatedMb_pct25",
    "AverageAllocatedMb_pct50",
    "AverageAllocatedMb_pct75",
    "AverageAllocatedMb_pct95",
    "AverageAllocatedMb_pct99",
    "AverageAllocatedMb_pct100"
]

plt.figure()
for col in percentile_cols:
    plt.plot(pdf[col], label=col)

plt.title("Memory Allocation Percentiles")
plt.xlabel("Row Index")
plt.ylabel("MB")
plt.legend()
plt.savefig("plot_memory_percentiles.png")

# 3. Heavy hitters by HashApp
top_apps = pdf["HashApp"].value_counts().head(10)

plt.figure()
plt.bar(top_apps.index.astype(str), top_apps.values)
plt.xticks(rotation=45)
plt.title("Top 10 Apps by Sample Count")
plt.ylabel("Count")
plt.savefig("plot_heavy_hitters.png")

# 4. SampleCount distribution
plt.figure()
plt.hist(pdf["SampleCount"].dropna(), bins=40)
plt.title("Distribution of SampleCount")
plt.xlabel("SampleCount")
plt.ylabel("Frequency")
plt.savefig("plot_samplecount_hist.png")

print("All plots generated.")
