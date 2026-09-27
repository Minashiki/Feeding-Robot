"""Frozen phase and contact surface declarations shared by runtime and validation."""

PHASES = (
    "SELECT",
    "ACQUIRE",
    "TRANSPORT",
    "WAIT_READY",
    "APPROACH",
    "TRANSFER",
    "RETRACT",
    "RECOVER",
    "STOP",
)
FREE_PHASES = ("SELECT", "TRANSPORT", "WAIT_READY")
MOUTH_PHASES = ("APPROACH", "TRANSFER", "RETRACT")
GEAR_OF = {name: "FREE" for name in FREE_PHASES}
GEAR_OF["ACQUIRE"] = "ACQUIRE"
for _name in MOUTH_PHASES:
    GEAR_OF[_name] = "MOUTH"
GEAR_OF["RECOVER"] = "STOP"
GEAR_OF["STOP"] = "STOP"

BOWL_GEOMS = ("bowl_bottom", "bowl_back", "bowl_left", "bowl_right", "bowl_front")
PLATE_GEOMS = ("plate_bottom",)
MOUTH_GEOMS = ("jaw_lip",)

