import polars as pl
from pathlib import Path

data_dir = Path("../data")

# print(data_dir / "DSU-Dataset.xlsx")


orig_data = pl.read_excel(data_dir / "DSU-Dataset.xlsx")
orig_data.write_parquet(data_dir / "DSU-Dataset.parquet")