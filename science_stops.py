import numpy as np
import rasterio
import pandas as pd
import csv
import math

# ── 1. Load maps ─────────────────────────────────────────────
with rasterio.open("roughness_map.tif") as f:
    roughness = f.read(1).astype(float)
    transform = f.transform

with rasterio.open("slope_map.tif") as f:
    slope = f.read(1).astype(float)

with rasterio.open("cost_map.tif") as f:
    cost = f.read(1).astype(float)

rows, cols = roughness.shape

# ── 2. Coordinate helpers ────────────────────────────────────
R = 1737400

def pixel_to_latlon(row, col, transform):
    x_m = transform.c + col * transform.a
    y_m = transform.f + row * transform.e
    lon = x_m / ((math.pi/180) * R)
    lat = y_m / ((math.pi/180) * R)
    return lat, lon

def latlon_to_pixel(lat, lon, transform):
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    col = int((x_m - transform.c) / transform.a)
    row = int((y_m - transform.f) / transform.e)
    return row, col

# ── 3. Landing point (rover start) ──────────────────────────
landing_lat, landing_lon = -85.60, 26.80
landing_row, landing_col = latlon_to_pixel(
    landing_lat, landing_lon, transform
)
print(f"Landing pixel: ({landing_row}, {landing_col})")

# ── 4. Normalize maps ────────────────────────────────────────
def normalize(arr):
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn)

roughness_n = normalize(roughness)
slope_n     = normalize(slope)

# ── 5. Science score map ─────────────────────────────────────
# Good science stop = moderate roughness + low slope + low cost
from scipy.ndimage import binary_dilation

# Crater edges = scientifically interesting
crater_mask = (cost > 0.7).astype(float)
crater_edges = binary_dilation(crater_mask, iterations=3).astype(float)
crater_edges[crater_mask > 0] = 0  # exclude crater interior

science_score = (
    0.35 * roughness_n +           # some roughness = interesting
    0.35 * crater_edges +           # near crater rim
    0.30 * (1 - slope_n)            # prefer gentle slopes
)

# Mask out dangerous terrain
science_score[cost > 0.80] = 0
science_score[slope_n > 0.7] = 0

# ── 6. Assign quadrants ──────────────────────────────────────
row_grid, col_grid = np.mgrid[0:rows, 0:cols]

quadrants = {
    "NE": (row_grid < landing_row) & (col_grid >= landing_col),
    "NW": (row_grid < landing_row) & (col_grid <  landing_col),
    "SE": (row_grid >= landing_row) & (col_grid >= landing_col),
    "SW": (row_grid >= landing_row) & (col_grid <  landing_col),
}

# ── 7. Find science stops per quadrant ──────────────────────
N_STOPS      = 10      # minimum stops per quadrant
SUPPRESS_PX  = 5       # suppression radius in pixels (~590m)

all_stops = []

for q_name, q_mask in quadrants.items():
    scores = science_score.copy()
    scores[~q_mask] = 0          # zero out other quadrants

    stops = []
    temp  = scores.copy()

    while len(stops) < N_STOPS:
        idx = np.unravel_index(np.argmax(temp), temp.shape)
        if temp[idx] == 0:
            print(f"  ⚠️  {q_name}: only {len(stops)} stops found")
            break

        lat, lon = pixel_to_latlon(idx[0], idx[1], transform)
        stops.append({
            "quadrant": q_name,
            "stop_id":  f"{q_name}_{len(stops)+1:02d}",
            "row":      idx[0],
            "col":      idx[1],
            "latitude": round(lat, 6),
            "longitude":round(lon, 6),
            "science_score": round(float(temp[idx]), 4),
            "slope":    round(float(slope_n[idx]), 4),
            "roughness":round(float(roughness_n[idx]), 4),
        })

        # Suppress neighborhood
        r0, c0 = idx
        r_min = max(0, r0 - SUPPRESS_PX)
        r_max = min(rows, r0 + SUPPRESS_PX)
        c_min = max(0, c0 - SUPPRESS_PX)
        c_max = min(cols, c0 + SUPPRESS_PX)
        temp[r_min:r_max, c_min:c_max] = 0

    all_stops.extend(stops)
    print(f"  ✅ {q_name}: {len(stops)} science stops found")

# ── 8. Save results ──────────────────────────────────────────
df = pd.DataFrame(all_stops)
df.to_csv("science_stops.csv", index=False)

print(f"\n✅ Total science stops: {len(all_stops)}")
print(f"✅ Saved to science_stops.csv")
print(f"\nSample stops:")
print(df[["stop_id","latitude","longitude",
          "science_score"]].to_string(index=False))