import polars as pl

# Load CSV
df = pl.read_csv(
    "app_memory_percentiles.anon.d01.csv",
    ignore_errors=True
)

# Drop rows with missing critical fields
df = df.drop_nulls(["HashOwner", "HashApp", "SampleCount"])

# Convert numeric columns to proper types
numeric_cols = [
    "SampleCount",
    "AverageAllocatedMb",
    "AverageAllocatedMb_pct1",
    "AverageAllocatedMb_pct5",
    "AverageAllocatedMb_pct25",
    "AverageAllocatedMb_pct50",
    "AverageAllocatedMb_pct75",
    "AverageAllocatedMb_pct95",
    "AverageAllocatedMb_pct99",
    "AverageAllocatedMb_pct100"
]

df = df.with_columns([
    pl.col(col).cast(pl.Float64, strict=False) for col in numeric_cols
])

# Save cleaned dataset
df.write_csv("app_memory_percentiles_clean.csv")

print("Cleaned dataset saved as app_memory_percentiles_clean.csv")
