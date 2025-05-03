from data import Task, Vehicle
import logging

logging.basicConfig(level=logging.DEBUG)

def pam_algorithm(tasks, vehicles, station_id, current_time):
    logging.debug(f"PAM: station_id={station_id}, current_time={current_time}")
    logging.debug(f"Tasks: {[t.__dict__ for t in tasks]}")
    logging.debug(f"Vehicles: {[v.__dict__ for v in vehicles]}")
    
    assignments = []
    available_vehicles = [v for v in vehicles if v.station == station_id and v.electricity >= 20]
    logging.debug(f"Available vehicles at {station_id}: {[v.id for v in available_vehicles]}")
    
    unassigned_tasks = [t for t in tasks if not t.assigned and t.origin == station_id]
    logging.debug(f"Unassigned tasks from {station_id}: {[t.__dict__ for t in unassigned_tasks]}")
    
    for vehicle in available_vehicles:
        vehicle_tasks = []
        total_fee = 0
        for task in unassigned_tasks:
            if (current_time + task.service_time <= task.deadline * 60 and
                vehicle.electricity >= task.service_time * 2 and
                len(vehicle_tasks) < vehicle.capacity):
                vehicle_tasks.append(task)
                total_fee += task.fee
                task.assigned = True
        if vehicle_tasks:
            assignments.append(f"{vehicle.id} -> {len(vehicle_tasks)} tasks to {vehicle_tasks[0].dest} (Fees: {total_fee})")
            logging.debug(f"Assigned: {vehicle.id} -> {vehicle_tasks}")
    
    return assignments