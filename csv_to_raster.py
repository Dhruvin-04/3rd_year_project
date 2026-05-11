import numpy as np
import rasterio
import pandas as pd
import math
from rasterio.features import rasterize
from shapely.geometry import Point

R = 1737400

with rasterio.open("slope_map.tif") as src:
    transform = src.transform
    meta      = src.meta.copy()
    height    = src.height
    width     = src.width

df = pd.read_csv("craters_latlong.csv")

# Convert lat/lon to Moon SimpleCylindrical meters
def latlon_to_meters(lat, lon):
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    return x_m, y_m

# Buffer radius ~500m per crater
CRATER_RADIUS_M = 500

shapes = []
for _, row in df.iterrows():
    x_m, y_m = latlon_to_meters(row['latitude'], row['longitude'])
    pt = Point(x_m, y_m)
    shapes.append((pt.buffer(CRATER_RADIUS_M).__geo_interface__, 1))

crater_mask = rasterize(
    shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype='uint8'
)

meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
with rasterio.open("crater_mask.tif", "w", **meta) as dst:
    dst.write(crater_mask, 1)

flagged = int(crater_mask.sum())
print(f"✅ crater_mask.tif saved")
print(f"   Crater pixels flagged: {flagged}")
print(f"   Coverage: {flagged/(height*width)*100:.2f}% of map")