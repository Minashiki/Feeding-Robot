"""Frozen M2 full matrix. Case identity determines stimuli, never trace metadata."""
from dataclasses import asdict, dataclass
import numpy as np

SCHEMA = "m2-full-v3"
VARIANTS = {"A": (.001, 50, 1e-8), "B": (.0005, 50, 1e-8), "C": (.001, 100, 1e-9)}
FORMAL_SEEDS = tuple(range(3, 9))
CALIBRATION_SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class Case:
    family: str
    pose: int = 0
    axis: int = 0
    sign: int = 1
    load: float = 0.
    stiffness: float = 500.
    delay: float = 0.
    noise: bool = False
    event: str = ""
    context: str = "free"

    @property
    def fixture(self):
        return self.family in {"spring","press","wall"} or self.context == "spring"

    @property
    def key(self):
        return (f"{self.family}-p{self.pose}-a{self.axis}-d{self.sign}-f{self.load:g}"
                f"-k{self.stiffness:g}-delay{self.delay:g}-noise{int(self.noise)}-{self.event or 'none'}-{self.context}")

    @property
    def duration(self):
        return {"hold": 6., "step": 5., "circle": 10., "sine": 10., "static": 3.,
                "loaded_circle": 10., "spring": 10., "stiffness": 5., "press": 10.,
                "wall": 6., "pulse": 5., "stop": 3., "carry": 7.,
                "plate": 30., "mouth": 30.}[self.family]

    def manifest(self, seed, variant):
        return {**asdict(self), "case_id": f"{self.key}-s{seed}-{variant}", "seed": seed,
                "variant": variant, "duration": self.duration,
                "dt": VARIANTS[variant][0], "iterations": VARIANTS[variant][1],
                "tolerance": VARIANTS[variant][2]}


def cases():
    result = []
    for pose in range(3):
        result.append(Case("hold", pose))
        result.append(Case("circle", pose))
        for axis in range(3):
            result.append(Case("sine", pose, axis))
            result.append(Case("loaded_circle", pose, axis, load=1.))
        for axis in range(6):
            for sign in (-1, 1):
                result.append(Case("step", pose, axis, sign))
                result.append(Case("pulse", pose, axis, sign))
                result.append(Case("stop", pose, axis, sign, event="expired"))
        for event in ("STOP", "RECOVER", "wrench", "contact", "nan", "inf", "warning", "speed", "penetration"):
            result.append(Case("stop", pose, event=event))
    for pose in range(6):
        result.append(Case("static", pose))
        for axis in range(6):
            for sign in (-1, 1):
                result.append(Case("static", pose, axis, sign, load=1. if axis < 3 else .05))
        result.append(Case("static", pose, axis=1, load=1., event="lever"))
    for pose in (0, 1):
        for stiffness in (200., 500., 1000.):
            result.append(Case("spring", pose, stiffness=stiffness))
        for gain in (150., 450.):
            result.append(Case("stiffness", pose, load=1., stiffness=gain))
            result.append(Case("press", pose, stiffness=gain))
        result.append(Case("wall", pose))
    for seed_index in range(3):
        for family in ("circle", "spring"):
            for delay in (0., .005, .01):
                for noise in (False, True):
                    # Pose encodes three separate noise/replay seeds; contact uses two legal poses.
                    result.append(Case(family, seed_index % 2 if family == "spring" else seed_index,
                                       delay=delay, noise=noise, event=f"sensor{seed_index}"))
    result.extend(Case(family) for family in ("carry", "plate", "mouth"))
    result.extend(Case("stop",pose,axis=1,sign=-1,event="expired",context="spring") for pose in (0,1))
    result.append(Case("stop",event="expired",context="food"))
    result.append(Case("stop",event="singularity"))
    assert len({c.key for c in result}) == len(result)
    return result


def manifest(seeds=FORMAL_SEEDS):
    return [case.manifest(seed, variant) for case in cases() for seed in seeds for variant in VARIANTS]

SINGULAR_Q = [-1.8729622292175738, 1.2953719117084952, .2528651848336265, -.3794389338812425, -.13147233171490358, 1.608251115930711, 1.6627727807805979]
REST_DQ = [0., .03, 0., 0., 0., 0., 0.]


TAU_MAX = np.array([87., 87., 87., 87., 12., 12., 12.])
RANGES = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973],
                   [-3.0718, -.0698], [-2.8973, 2.8973], [-.0175, 3.7525], [-2.8973, 2.8973]])
GEARS = {"FREE": (.05, .3, .02, .13962634, 8., .8),
         "ACQUIRE": (.02, .15, .01, .08726646, 4., .4),
         "MOUTH": (.01, .1, .005, .05235988, 2., .2),
         "STOP": (0., 0., .02, .13962634, 8., .8)}


BLEND_S = 0.2
BLEND_TIME_TOL_S = 1e-12

GEAR_K = {
    "FREE": np.array([300.0, 300.0, 300.0, 8.0, 8.0, 8.0]),
    "ACQUIRE": np.array([150.0, 150.0, 150.0, 5.0, 5.0, 5.0]),
    "MOUTH": np.array([100.0, 100.0, 100.0, 3.0, 3.0, 3.0]),
    "STOP": np.zeros(6),
}
GEAR_D = {
    "FREE": np.array([49.0, 49.0, 49.0, 0.8, 0.8, 0.8]),
    "ACQUIRE": np.array([35.0, 35.0, 35.0, 0.63, 0.63, 0.63]),
    "MOUTH": np.array([28.0, 28.0, 28.0, 0.49, 0.49, 0.49]),
    "STOP": np.zeros(6),
}
GEAR_VW = {"FREE": (0.05, 0.3), "ACQUIRE": (0.02, 0.15), "MOUTH": (0.01, 0.1), "STOP": (0.0, 0.0)}


