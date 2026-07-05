TACTICAL_ZONES = {
    "GK",
    "LB",
    "LCB",
    "CB",
    "RCB",
    "RB",
    "LWB",
    "RWB",
    "DM",
    "LCM",
    "CM",
    "RCM",
    "AM",
    "LW",
    "RW",
    "ST",
    "LST",
    "RST",
}

PREFERRED_SIDES = {
    "left",
    "center",
    "right",
    "either",
    "unknown",
}


def normalize_zone(zone: str | None) -> str | None:
    if zone is None:
        return None
    normalized = zone.strip().upper()
    return normalized or None


def validate_zone(zone: str | None) -> str | None:
    normalized = normalize_zone(zone)
    if normalized is None:
        return None
    if normalized not in TACTICAL_ZONES:
        raise ValueError(
            f"zone must be one of: {sorted(TACTICAL_ZONES)}"
        )
    return normalized


def validate_zones(zones: list[str] | None) -> list[str]:
    if not zones:
        return []
    return [
        zone
        for zone in [
            validate_zone(item)
            for item in zones
        ]
        if zone is not None
    ]
