"""Prep step: download yolo11l.pt next to the code (needs internet). Run once before the timed run."""
import os
from pathlib import Path

os.chdir(Path(__file__).resolve().parent)   # ultralytics saves into the current folder
from ultralytics import YOLO

YOLO("yolo11l.pt")
print("weights ready:", Path("yolo11l.pt").resolve(), Path("yolo11l.pt").stat().st_size, "bytes")
