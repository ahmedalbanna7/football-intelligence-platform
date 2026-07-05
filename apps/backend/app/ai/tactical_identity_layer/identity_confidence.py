def clamp_score(value: float) -> float:
    return round(max(0.0, min(value, 1.0)), 4)


def weighted_average(weights: dict[str, float], values: dict[str, float | None]) -> float:
    total_weight = 0.0
    total = 0.0
    for key, weight in weights.items():
        value = values.get(key)
        if value is None:
            continue
        total_weight += weight
        total += value * weight

    if total_weight == 0:
        return 0.0
    return clamp_score(total / total_weight)
