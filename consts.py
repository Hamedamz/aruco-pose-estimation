from enum import Enum


camera_map = {
    "pi3": 0,
    "pi3w": 1,
    "pihq6mm": 0,
}

res_map = {
    "244p": (244, 324),
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
}

class Marker(Enum):
    ARUCO = 1
    STAG = 2
    APRILTAG = 3
    P4 = 4

