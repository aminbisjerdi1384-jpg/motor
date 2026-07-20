"""
bearing.py
----------
Bearing characteristic fault frequency calculations.

A bearing has 4 well-known characteristic defect frequencies, each a
multiple of the shaft rotation frequency fr = RPM / 60:

    BPFO  - Ball Pass Frequency, Outer race
    BPFI  - Ball Pass Frequency, Inner race
    BSF   - Ball Spin Frequency  (rolling element defect)
    FTF   - Fundamental Train Frequency (cage defect)

Given bearing geometry (n balls, ball diameter d, pitch diameter D,
contact angle theta), the standard formulas are:

    BPFO = (n/2) * fr * (1 - (d/D) cos(theta))
    BPFI = (n/2) * fr * (1 + (d/D) cos(theta))
    BSF  = (D/(2d)) * fr * (1 - (d/D)^2 cos(theta)^2)
    FTF  = (fr/2)  * (1 - (d/D) cos(theta))

For the CWRU dataset specifically, Case Western Reserve University
publishes the resulting frequencies directly as multiples of running
speed (verified against
https://engineering.case.edu/bearingdatacenter/bearing-information),
which avoids any ambiguity about the exact contact angle used. Those
official multipliers are used by default for CWRU data; the geometric
formulas above are provided for any other bearing/motor.
"""
from dataclasses import dataclass
from math import cos, radians
from typing import Optional


@dataclass
class BearingGeometry:
    n_balls: float
    ball_diameter: float      # d
    pitch_diameter: float     # D
    contact_angle_deg: float = 0.0


# Official CWRU Bearing Data Center specs (inches), drive-end (6205-2RS JEM
# SKF) and fan-end (6203-2RS JEM SKF) deep-groove ball bearings.
CWRU_DRIVE_END_GEOMETRY = BearingGeometry(
    n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537, contact_angle_deg=0.0
)
CWRU_FAN_END_GEOMETRY = BearingGeometry(
    n_balls=9, ball_diameter=0.2656, pitch_diameter=1.122, contact_angle_deg=0.0
)

# Official CWRU-published defect frequencies, as a multiple of running
# speed (Hz). These are the authoritative numbers to use for CWRU data.
CWRU_DRIVE_END_MULTIPLIERS = {
    "BPFI": 5.4152,
    "BPFO": 3.5848,
    "FTF": 0.39828,
    "BSF": 4.7135,
}
CWRU_FAN_END_MULTIPLIERS = {
    "BPFI": 4.9469,
    "BPFO": 3.0530,
    "FTF": 0.3817,
    "BSF": 3.9874,
}


def rotational_frequency_hz(rpm: float) -> float:
    """fr = RPM / 60."""
    return float(rpm) / 60.0


def fault_frequencies_from_geometry(rpm: float, geometry: BearingGeometry) -> dict:
    """
    Compute BPFO, BPFI, BSF, FTF (in Hz) from first principles, given
    bearing geometry and shaft speed.
    """
    fr = rotational_frequency_hz(rpm)
    n = geometry.n_balls
    d = geometry.ball_diameter
    D = geometry.pitch_diameter
    cos_theta = cos(radians(geometry.contact_angle_deg))
    ratio = d / D

    bpfo = (n / 2.0) * fr * (1 - ratio * cos_theta)
    bpfi = (n / 2.0) * fr * (1 + ratio * cos_theta)
    bsf = (D / (2.0 * d)) * fr * (1 - (ratio ** 2) * (cos_theta ** 2))
    ftf = (fr / 2.0) * (1 - ratio * cos_theta)

    return {"fr": fr, "BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


def fault_frequencies_from_multipliers(rpm: float, multipliers: dict) -> dict:
    """
    Compute BPFO, BPFI, BSF, FTF (in Hz) using pre-published multipliers
    of running speed (the CWRU approach).
    """
    fr = rotational_frequency_hz(rpm)
    return {
        "fr": fr,
        "BPFO": fr * multipliers["BPFO"],
        "BPFI": fr * multipliers["BPFI"],
        "BSF": fr * multipliers["BSF"],
        "FTF": fr * multipliers["FTF"],
    }


def cwru_fault_frequencies(rpm: float, position: str = "drive_end") -> dict:
    """
    Convenience function: official CWRU fault frequencies for the given
    shaft speed and bearing position ('drive_end' or 'fan_end').
    """
    multipliers = (
        CWRU_DRIVE_END_MULTIPLIERS if position == "drive_end" else CWRU_FAN_END_MULTIPLIERS
    )
    return fault_frequencies_from_multipliers(rpm, multipliers)


def label_to_target_frequency(label: str, fault_freqs: dict) -> Optional[float]:
    """
    Map a fault class label (as used throughout this app) to the
    corresponding theoretical fault frequency, for peak-matching against
    a measured (envelope) spectrum.
    """
    mapping = {
        "Inner Race Fault": fault_freqs.get("BPFI"),
        "Outer Race Fault": fault_freqs.get("BPFO"),
        "Ball Fault": fault_freqs.get("BSF"),
        "Cage Fault": fault_freqs.get("FTF"),
    }
    return mapping.get(label)
