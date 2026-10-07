"""Evaluate image sparse_rcnn on the test split only."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.runner import main
from detectors.sparse_rcnn.build import build_model

if __name__ == "__main__":
    main("evaluate", task="image", version="sparse_rcnn", build_model=build_model)
