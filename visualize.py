import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

def visualize_results(stations, vehicles, tasks, assignments, repositioning):
    fig, ax = plt.subplots()
    
    station_coords = {s.id: s.location for s in stations}
    station_points = {s.id: s.charging_points for s in stations}
    
    for s_id, (x, y) in station_coords.items():
        ax.scatter(x, y, c='blue', s=100, label=s_id if s_id == list(station_coords.keys())[0] else "")
        ax.text(x, y+0.1, f"{s_id} ({station_points[s_id]})")
    
    vehicle_positions = {v.id: v.station for v in vehicles}
    
    def update(frame):
        ax.clear()
        for s_id, (x, y) in station_coords.items():
            ax.scatter(x, y, c='blue', s=100)
            ax.text(x, y+0.1, f"{s_id} ({station_points[s_id]})")
        
        if frame < len(assignments):
            assignment = assignments[frame]
            vehicle_id = assignment.split()[0]
            dest = assignment.split()[-3]
            origin = next(v.station for v in vehicles if v.id == vehicle_id)
            x1, y1 = station_coords[origin]
            x2, y2 = station_coords[dest]
            ax.arrow(x1, y1, x2-x1, y2-y1, color='blue', head_width=0.1, length_includes_head=True)
            ax.text((x1+x2)/2, (y1+y2)/2, vehicle_id)
        
        if frame < len(repositioning):
            reposition = repositioning[frame]
            vehicle_id = reposition.split()[0]
            origin = reposition.split()[3]
            dest = reposition.split()[-1]
            x1, y1 = station_coords[origin]
            x2, y2 = station_coords[dest]
            ax.arrow(x1, y1, x2-x1, y2-y1, color='red', head_width=0.1, length_includes_head=True)
            ax.text((x1+x2)/2, (y1+y2)/2, vehicle_id)
    
    ani = FuncAnimation(fig, update, frames=max(len(assignments), len(repositioning)), repeat=False)
    plt.legend()
    plt.show()