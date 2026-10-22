"""Physical metrics independent of reward weighting."""
import numpy as np


def summarize(records, step_minutes):
    dt = step_minutes / 60
    occupancy = np.array([r["occupancy"] for r in records])
    occupied = occupancy > 0
    violation = np.array([r["violation"] for r in records])
    starts = np.array([r["class_start"] for r in records])
    start_window = np.zeros_like(starts)
    for t, room in zip(*np.nonzero(starts)):
        start_window[t:min(len(starts), t + int(np.ceil(15 / step_minutes))), room] = True
    start_window &= occupied
    room_hours = occupied.sum(axis=0) * dt
    room_discomfort = (violation * occupied).sum(axis=0) * dt
    room_mean = np.divide(room_discomfort, room_hours, out=np.zeros(2), where=room_hours > 0)
    present = room_hours > 0
    student_hours = float(occupancy.sum() * dt)
    return {
        "energy_kwh": float(sum(r["energy_kwh"] for r in records)),
        "occupied_room_degree_hours": float(room_discomfort.sum()),
        "student_degree_hours": float((violation * occupancy).sum() * dt),
        "mean_student_discomfort_c": float((violation * occupancy).sum() * dt / student_hours) if student_hours else 0,
        "comfort_violation_pct": float(100 * np.sum((violation > 0) & occupied) / max(occupied.sum(), 1)),
        "class_start_violation_pct": float(100 * np.sum((violation > 0) & start_window) / max(start_window.sum(), 1)),
        "room_0_mean_discomfort_c": float(room_mean[0]),
        "room_1_mean_discomfort_c": float(room_mean[1]),
        "worst_room_mean_discomfort_c": float(max(room_mean)),
        "room_discomfort_gap_c": float(np.ptp(room_mean[present])) if present.any() else 0,
        "ac_switches": int(sum(r["switches"] for r in records)),
        "cap_violations": int(sum(r["power_kw"] > r["cap_kw"] + 1e-9 for r in records)),
    }
