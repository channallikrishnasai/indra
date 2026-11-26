"""Extract a handful of named initial-condition windows for the DASHBOARD (not used by the server).

Each window = 3 input days + 10 observed days (to show model-vs-reality verification in the UI).
Writes dashboard/public/windows.bin (float32) + windows.json (metadata).
    python tools/make_windows.py --data data/india_climate_train.nc
"""
import argparse, json
from pathlib import Path
import numpy as np
import xarray as xr

EVENTS = [  # (last input day, title, short description)
    ("2021-06-02", "Monsoon onset over Kerala", "Southwest monsoon arrives on India's southern tip"),
    ("2021-05-14", "Cyclone Tauktae", "Extremely severe cyclone tracking up the Arabian Sea coast"),
    ("2021-05-23", "Cyclone Yaas", "Very severe cyclone heading for Odisha / West Bengal"),
    ("2019-05-01", "Cyclone Fani", "Extremely severe cyclone approaching Odisha"),
    ("2019-06-01", "North India heatwave", "Peak pre-monsoon heat over Rajasthan and the plains"),
    ("2018-08-13", "Kerala floods", "Extreme monsoon rain over the Western Ghats"),
    ("2021-07-20", "Peak monsoon (Konkan floods)", "Active monsoon with very heavy rain on the west coast"),
    ("2021-01-18", "Winter western disturbance", "Mid-latitude trough bringing rain and snow to the north"),
    ("2021-10-15", "Monsoon withdrawal", "Retreating monsoon, northeast monsoon setting in"),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/india_climate_train.nc")
    ap.add_argument("--out", default="dashboard/public"); a = ap.parse_args()
    ds = xr.open_dataset(a.data); times = ds["time"].values.astype("datetime64[D]")
    blocks, meta = [], []
    for day, title, desc in EVENTS:
        i = int(np.where(times == np.datetime64(day))[0][0])
        blocks.append(ds["data"].values[i - 2:i + 11].astype("f4"))      # 3 inputs + 10 observed
        doy = int((np.datetime64(day) - np.datetime64(day[:4] + "-01-01")).astype(int)) + 1
        meta.append({"id": day, "title": title, "desc": desc, "date": day, "doy": doy,
                     "doy_sincos": [float(np.sin(2 * np.pi * doy / 365)), float(np.cos(2 * np.pi * doy / 365))]})
    arr = np.stack(blocks)                                                  # [W,13,6,32,36]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    arr.astype("<f4").tofile(out / "windows.bin")
    (out / "windows.json").write_text(json.dumps({"shape": list(arr.shape), "windows": meta}, indent=1))
    print("wrote", arr.shape, (out / "windows.bin").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
