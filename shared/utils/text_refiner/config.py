"""Configuration for text-refiner client."""

import os

TEXT_REFINER_URL = os.getenv("TEXT_REFINER_URL", "http://text-refiner:8000")
ENABLE_TEXT_REFINER = os.getenv("ENABLE_TEXT_REFINER", "true").lower() == "true"
TEXT_REFINER_TIMEOUT = float(os.getenv("TEXT_REFINER_TIMEOUT", "2.0"))
