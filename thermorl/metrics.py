"""Physical metrics independent of reward weighting."""
import numpy as np

METRIC_DEFINITIONS = {
    "version": "comfort_v2",
    "arrival": "Occupied explicit class starts, pre-action air temperature minus previously active fan offset; event weighted.",
    "early_class": "Post-action samples weighted by occupied overlap with the first 15 clock minutes after a start; windows stop at class/empty/day boundaries, unioned per room.",
    "whole_day": "Post-action comfortable occupied room-time; energy includes all equipment use, including pre-cooling.",
    "undefined": "Arrival percentage is null when undefined/unavailable. Legacy early/whole-day percentages retain 0 with explicit zero denominators; display Not applicable.",
}


def pooled_comfort_metrics(rows):
    """Pool event/time counts, never unweighted daily percentages."""
    def total(key):
        return sum(float(r[key]) for r in rows if r.get(key) not in (None, ''))

    arrivals, measured = total('class_arrivals'), total('arrival_measurements')
    early, occupied = total('early_class_occupied_minutes'), total('occupied_room_minutes')
    return {
        'arrival_discomfort_pct': 100 * total('uncomfortable_arrivals') / arrivals if arrivals and measured == arrivals else None,
        'class_start_violation_pct': 100 * total('early_class_uncomfortable_minutes') / early if early else None,
        'comfort_pct': 100 * total('comfortable_room_minutes') / occupied if occupied else None,
    }


def summarize(records, step_minutes):
    dt = step_minutes / 60
    occupancy = np.array([r["occupancy"] for r in records])
    occupied = occupancy > 0
    violation = np.array([r["violation"] for r in records])
    count = max(int(occupied.sum()), 1)
    hot = np.array([r.get("overheated", np.zeros(2, dtype=bool)) for r in records])
    cold = np.array([r.get("overcooled", np.zeros(2, dtype=bool)) for r in records])
    starts = np.array([r["class_start"] for r in records]) & occupied
    early_minutes = np.zeros_like(violation, dtype=float)
    remaining = np.zeros(occupancy.shape[1])
    for t in range(len(records)):
        remaining[~occupied[t]] = 0
        remaining[starts[t]] = 15
        early_minutes[t] = np.minimum(remaining, step_minutes) * occupied[t]
        remaining = np.maximum(remaining - step_minutes, 0)
    early_total = float(early_minutes.sum())
    early_uncomfortable = float((early_minutes * (violation > 0)).sum())
    arrivals = int(starts.sum())
    measured = uncomfortable = 0
    for record, events in zip(records, starts):
        if 'arrival_violation' in record:
            measured += int(events.sum())
            uncomfortable += int(np.sum(events & (np.asarray(record['arrival_violation']) > 0)))
    room_hours = occupied.sum(axis=0) * dt
    room_discomfort = (violation * occupied).sum(axis=0) * dt
    room_mean = np.divide(room_discomfort, room_hours, out=np.zeros(2), where=room_hours > 0)
    present = room_hours > 0
    student_hours = float(occupancy.sum() * dt)
    return {
        "energy_kwh": float(sum(r["energy_kwh"] for r in records)),
        "class_arrivals": arrivals, "arrival_measurements": measured,
        "uncomfortable_arrivals": uncomfortable if measured == arrivals else None,
        "arrival_discomfort_pct": 100 * uncomfortable / arrivals if arrivals and measured == arrivals else None,
        "early_class_occupied_minutes": early_total,
        "early_class_uncomfortable_minutes": early_uncomfortable,
        "occupied_room_minutes": float(occupied.sum() * step_minutes),
        "comfortable_room_minutes": float(np.sum((violation == 0) & occupied) * step_minutes),
        "occupied_room_degree_hours": float(room_discomfort.sum()),
        "student_degree_hours": float((violation * occupancy).sum() * dt),
        "mean_student_discomfort_c": float((violation * occupancy).sum() * dt / student_hours) if student_hours else 0,
        "comfort_violation_pct": float(100 * np.sum((violation > 0) & occupied) / max(occupied.sum(), 1)),
        "comfort_pct": float(100 * np.sum((violation == 0) & occupied) / count),
        "overheating_pct": float(100 * np.sum(hot & occupied) / count),
        "overcooling_pct": float(100 * np.sum(cold & occupied) / count),
        "class_start_violation_pct": 100 * early_uncomfortable / early_total if early_total else 0.0,
        "room_0_mean_discomfort_c": float(room_mean[0]),
        "room_1_mean_discomfort_c": float(room_mean[1]),
        "worst_room_mean_discomfort_c": float(max(room_mean)),
        "room_discomfort_gap_c": float(np.ptp(room_mean[present])) if present.any() else 0,
        "ac_switches": int(sum(r["switches"] for r in records)),
        "cap_violations": int(sum(r["power_kw"] > r["cap_kw"] + 1e-9 for r in records)),
    }
