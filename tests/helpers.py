import numpy as np
import pandas as pd


def make_toy_cmapss_df(units: int, cycles: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for unit_id in range(1, units + 1):
        t = np.arange(1, cycles + 1, dtype="int64")
        base = np.sin(t / 3.0) + np.cos(t / 5.0)
        for cycle in t:
            row = {
                "unit_id": unit_id,
                "cycle": int(cycle),
                "setting1": float(base[cycle - 1] + rng.normal(0, 0.05)),
                "setting2": float(base[cycle - 1] * 0.5 + rng.normal(0, 0.05)),
                "setting3": float(base[cycle - 1] * -0.25 + rng.normal(0, 0.05)),
            }
            for i in range(1, 22):
                signal = base[cycle - 1] * (1 + i * 0.01)
                row[f"s{i}"] = float(signal + rng.normal(0, 0.1))
            row["rul"] = int(cycles - cycle)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df
