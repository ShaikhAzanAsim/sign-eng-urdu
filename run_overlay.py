"""Convenience entry point so PyCharm / `python run_overlay.py` works without PYTHONPATH tricks."""

from src.landmark_extraction import main

if __name__ == "__main__":
    raise SystemExit(main())
