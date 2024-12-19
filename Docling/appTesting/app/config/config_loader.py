import json
from pathlib import Path

def load_config(config_path):
    """
    Load configuration from the JSON file.
    If the file doesn't exist, return a default configuration.
    """
    if Path(config_path).exists():
        with open(config_path) as f:
            return json.load(f)
    else:
        # Default configuration if file doesn't exist
        return {
            "output_directory": r"..\data\IngestedFiles",
            "temp_directory": r"..\data\TempFiles",
            "mapping_file": r"..\client_mapping.json",
            "supported_formats": [".pdf", ".docx", ".xlsx", ".odt", ".ods", ".png", ".tiff"]
        }
