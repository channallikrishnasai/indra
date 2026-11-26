"""Per-lead rain-map strength. Run from the repo root:  python tests/test_qmix_schedule.py
Checks: a scalar qmix equals the same value repeated per lead (so existing checkpoints are unchanged);
a per-lead list applies each value to its own lead and nothing else."""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from indra_model.net import load_indra  # noqa: E402
import xarray as xr  # noqa: E402

m = load_indra(ROOT / "weights" / "indra.pt")
ds = xr.open_dataset(ROOT / "data" / "india_climate_train.nc")
data = ds["data"].values.astype("f4"); t = ds["time"].values.astype("datetime64[D]")
i = int(np.where(t == np.datetime64("2021-07-15"))[0][0]); init = data[i-2:i+1]; doy = float((t - t.astype("datetime64[Y]")).astype(int)[i])
m.qmix = 0.5; a = m.simulate(init, doy, 10)
m.qmix = [0.5] * 10; b = m.simulate(init, doy, 10)
print("scalar 0.5 vs list of 0.5 (max abs diff):", float(np.abs(a - b).max()))
assert np.array_equal(a, b)
m.qmix = [0.5] * 5 + [0.25] * 5; c = m.simulate(init, doy, 10)
print("per-lead list, leads 1-5 identical to 0.5:", bool(np.array_equal(c[:5, 1], a[:5, 1])))
print("per-lead list, leads 6-10 differ from 0.5:", bool(not np.array_equal(c[5:, 1], a[5:, 1])))
assert np.array_equal(c[:5], a[:5]) and not np.array_equal(c[5:, 1], a[5:, 1])
# other channels are untouched by the rain map
assert np.array_equal(c[:, [0, 2, 3, 4, 5]], a[:, [0, 2, 3, 4, 5]])
print("QMIX SCHEDULE TESTS PASSED")
