import os
import random
import json
from pathlib import Path
from werkzeug.utils import secure_filename
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {
        "output_directory": r"data\IngestedFiles",
        "temp_directory": r"data\TempFiles",
        "mapping_file": r"client_mapping.json",
        "supported_formats": [".pdf", ".docx", ".xlsx", ".odt", ".ods", ".png", ".tiff"]
    }

config = load_config()

def load_client_mapping():
    """Load client mapping."""
    mapping_file = config["mapping_file"]
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            return json.load(f), mapping_file
    return {}, mapping_file

def save_client_mapping(client_mapping, mapping_file):
    """Save client mapping."""
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)

def generate_client_id(client_mapping):
    """Generate a unique 8-digit Client ID."""
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            return client_id

def validate_file_format(filename):
    """Check if the file has a supported format."""
    _, ext = os.path.splitext(filename.lower())
    return ext in config["supported_formats"]

def save_temp_file(file):
    """Save the uploaded file to the temporary directory."""
    temp_path = Path(config["temp_directory"]) / secure_filename(file.filename)
    file.save(temp_path)
    return temp_path

def get_client_output_dir(client_id):
    """Get the output directory for a given Client ID."""
    return Path(config["output_directory"]) / client_id

def process_document(file_path, output_dir, global_client_id):
    """Process the document using Docling."""
    # Your process_document logic here.
    pass
