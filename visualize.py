import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import math

# ── 1. Load data ─────────────────────────────────────────────
with rasterio.open("cost_map.tif") as f:
    cost = f.read(1).astype(float)
    transform = f.transform

with rasterio.open("hillshade_map.tif") as f:
    hillshade = f.read(1).astype(float)

path_df  = pd.read_csv("rover_path.csv")
stops_df = pd.read_csv("science_stops.csv")
crater_df= pd.read_csv("craters_latlong.csv")

R = 1737400

def latlon_to_pixel(lat, lon, transform):
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    col = (x_m - transform.c) / transform.a
    row = (y_m - transform.f) / transform.e
    return row, col

# ── 2. Convert all coords to pixels ─────────────────────────
# Path
path_rows = path_df['row'].values
path_cols = path_df['col'].values

# Science stops
stops_df['px_row'], stops_df['px_col'] = zip(*[
    latlon_to_pixel(r.latitude, r.longitude, transform)
    for _, r in stops_df.iterrows()
])

# Craters
crater_df['px_row'], crater_df['px_col'] = zip(*[
    latlon_to_pixel(r.latitude, r.longitude, transform)
    for _, r in crater_df.iterrows()
])

# ── 3. Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 10))
fig.patch.set_facecolor('#1a1a2e')

quadrant_colors = {
    'NE': '#00ff88',
    'NW': '#00aaff', 
    'SE': '#ffaa00',
    'SW': '#ff4488'
}

for ax, (bg, bg_label) in zip(axes, 
    [(hillshade, 'Hillshade'), (cost, 'Cost Map')]):

    # Background
    if bg_label == 'Hillshade':
        ax.imshow(bg, cmap='gray', origin='upper',
                  vmin=np.nanpercentile(bg,2),
                  vmax=np.nanpercentile(bg,98))
    else:
        cmap = LinearSegmentedColormap.from_list(
            'safe', ['#00ff00','#ffff00','#ff0000'])
        im = ax.imshow(bg, cmap=cmap, origin='upper',
                       vmin=0, vmax=1, alpha=0.85)
        plt.colorbar(im, ax=ax, label='Cost (0=safe, 1=dangerous)',
                     fraction=0.03)

    # Crater detections
    ax.scatter(crater_df['px_col'], crater_df['px_row'],
               c='cyan', s=8, alpha=0.5,
               label='Detected Craters', zorder=3)

    # Rover path
    ax.plot(path_cols, path_rows,
            'w-', linewidth=2, zorder=4, label='Rover Path')
    ax.plot(path_cols[0], path_rows[0],
            'w^', markersize=10, zorder=5, label='Start')
    ax.plot(path_cols[-1], path_rows[-1],
            'w*', markersize=12, zorder=5, label='End')

    # Science stops by quadrant
    for q, color in quadrant_colors.items():
        q_stops = stops_df[stops_df['quadrant'] == q]
        ax.scatter(q_stops['px_col'], q_stops['px_row'],
                   c=color, s=60, marker='D', 
                   edgecolors='white', linewidths=0.5,
                   zorder=6, label=f'{q} stops')

    # Quadrant dividers
    ax.axhline(y=217, color='white', linewidth=0.5,
               linestyle='--', alpha=0.4)
    ax.axvline(x=335, color='white', linewidth=0.5,
               linestyle='--', alpha=0.4)

    ax.set_title(f'Lunar Rover Navigation — {bg_label}',
                 color='white', fontsize=13, pad=10)
    ax.set_xlabel('Column (pixels)', color='white')
    ax.set_ylabel('Row (pixels)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')

    if bg_label == 'Hillshade':
        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#1a1a2e', labelcolor='white')

plt.suptitle(
    'Chandrayaan-2 OHRC/LOLA — South Polar Rover Navigation\n'
    'Landing: 85.6°S 26.8°E  |  40 Science Stops  |  A* Path Planning',
    color='white', fontsize=14, y=1.01
)

plt.tight_layout()
plt.savefig("rover_navigation_final.png",
            dpi=150, bbox_inches='tight',
            facecolor='#1a1a2e')
plt.show()
print("✅ Saved rover_navigation_final.png")