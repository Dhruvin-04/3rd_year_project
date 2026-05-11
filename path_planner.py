import numpy as np
import rasterio
import heapq
import csv

# ── 1. Load cost map ─────────────────────────────────────────
with rasterio.open("cost_map.tif") as src:
    cost = src.read(1).astype(float)
    transform = src.transform
    meta = src.meta.copy()

rows, cols = cost.shape

# ── 2. Coordinate helpers ────────────────────────────────────
def latlon_to_pixel(lat, lon, transform):
    """Convert lat/lon degrees to pixel row, col"""
    import math
    R = 1737400
    x_m = lon * (math.pi/180) * R
    y_m = lat * (math.pi/180) * R
    col = int((x_m - transform.c) / transform.a)
    row = int((y_m - transform.f) / transform.e)
    return row, col

def pixel_to_latlon(row, col, transform):
    """Convert pixel row, col to lat/lon degrees"""
    import math
    R = 1737400
    x_m = transform.c + col * transform.a
    y_m = transform.f + row * transform.e
    lon = x_m / ((math.pi/180) * R)
    lat = y_m / ((math.pi/180) * R)
    return lat, lon

# ── 3. Define start and end points ──────────────────────────
# Using center of your crater detection region as start
# Adjust these to your desired start/end locations
start_latlon = (-85.60, 26.80)   # start point
end_latlon   = (-85.25, 26.95)   # end point

start = latlon_to_pixel(*start_latlon, transform)
end   = latlon_to_pixel(*end_latlon,   transform)

print(f"Start pixel: {start}")
print(f"End pixel:   {end}")
print(f"Cost at start: {cost[start]:.3f}")
print(f"Cost at end:   {cost[end]:.3f}")

# ── 4. A* Algorithm ──────────────────────────────────────────
def heuristic(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def astar(cost_map, start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    came_from = {}
    g_score = np.full(cost_map.shape, np.inf)
    g_score[start] = 0
    
    f_score = np.full(cost_map.shape, np.inf)
    f_score[start] = heuristic(start, end)
    
    # 8-directional movement
    neighbors = [(-1,-1),(-1,0),(-1,1),
                 ( 0,-1),       ( 0,1),
                 ( 1,-1),( 1,0),( 1,1)]
    
    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == end:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        
        for dr, dc in neighbors:
            nr, nc = current[0]+dr, current[1]+dc
            
            # Bounds check
            if not (0 <= nr < cost_map.shape[0] and
                    0 <= nc < cost_map.shape[1]):
                continue
            
            # Skip impassable terrain (cost = 1.0)
            if cost_map[nr, nc] >= 0.95:
                continue
            
            # Diagonal movement costs more
            move_cost = 1.414 if (dr != 0 and dc != 0) else 1.0
            tentative_g = g_score[current] + move_cost * cost_map[nr, nc]
            
            if tentative_g < g_score[nr, nc]:
                came_from[(nr, nc)] = current
                g_score[nr, nc] = tentative_g
                f_score[nr, nc] = tentative_g + heuristic((nr,nc), end)
                heapq.heappush(open_set, (f_score[nr,nc], (nr,nc)))
    
    return None  # No path found

# ── 5. Run A* ────────────────────────────────────────────────
print("\nRunning A* path planner...")
path = astar(cost, start, end)

if path is None:
    print("❌ No path found — try different start/end points")
else:
    print(f"✅ Path found: {len(path)} waypoints")
    
    # ── 6. Save path as CSV ──────────────────────────────────
    with open("rover_path.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "row", "col", "latitude", "longitude", "cost"])
        for i, (r, c) in enumerate(path):
            lat, lon = pixel_to_latlon(r, c, transform)
            writer.writerow([i, r, c, lat, lon, cost[r, c]])
    
    print(f"✅ Path saved to rover_path.csv")
    print(f"   Total pixels traversed: {len(path)}")
    print(f"   Start: {start_latlon}")
    print(f"   End:   {end_latlon}")