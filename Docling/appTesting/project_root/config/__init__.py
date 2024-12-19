import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    "output_directory": r"..\data\IngestedFiles",
    "temp_directory": r"..\data\TempFiles",
    "mapping_file": r"..\client_mapping.json",
    "supported_formats": [".pdf", ".docx", ".xlsx", ".odt", ".ods", ".png", ".tiff"],
}

def load_config(config_path="config.json"):
    """
    Load configuration from the JSON file.
    If the file doesn't exist, return the default configuration.
    """
    config_file = Path(config_path)
    if config_file.exists():
        with config_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return DEFAULT_CONFIG
