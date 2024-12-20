import os
import random
import json
import logging
from pathlib import Path
from werkzeug.utils import secure_filename
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"

def load_config():
    """Load configuration from the config file or return default configuration."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            logger.info("Loaded configuration file.")
            return json.load(f)
    logger.warning("Configuration file not found. Using defaults.")
    return {
        "output_directory": r"data\IngestedFiles",
        "temp_directory": r"data\TempFiles",
        "mapping_file": r"client_mapping.json",
        "supported_formats": [".pdf", ".docx", ".xlsx", ".odt", ".ods", ".png", ".tiff"]
    }

config = load_config()

def load_client_mapping():
    """Load client mapping from the mapping file or return an empty mapping."""
    mapping_file = config["mapping_file"]
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            logger.info("Loaded client mapping.")
            return json.load(f), mapping_file
    logger.warning("Client mapping file not found. Starting fresh.")
    return {}, mapping_file

def save_client_mapping(client_mapping, mapping_file):
    """Save client mapping to the mapping file."""
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)
        logger.info("Saved client mapping.")

def generate_client_id(client_mapping):
    """Generate a unique 8-digit Client ID."""
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            logger.info(f"Generated new Client ID: {client_id}")
            return client_id

def validate_file_format(filename):
    """Check if the file has a supported format."""
    _, ext = os.path.splitext(filename.lower())
    is_valid = ext in config["supported_formats"]
    logger.info(f"File format validation for {filename}: {'valid' if is_valid else 'invalid'}")
    return is_valid

def get_client_output_dir(client_id):
    """Get the output directory for a given Client ID."""
    output_dir = Path(config["output_directory"]) / client_id
    logger.info(f"Resolved output directory for Client ID {client_id}: {output_dir}")
    return output_dir

def process_document(file_path, output_dir, global_client_id):
    """Process the document using Docling."""
    logger.info(f"Starting document processing for {file_path}.")
    try:
        # Your existing `process_document` logic here.
        logger.info(f"Document processed successfully for {file_path}.")
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise
