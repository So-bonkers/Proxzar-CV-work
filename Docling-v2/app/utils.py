import os
import random
# import requests
from urllib.parse import urlparse
import json
import logging
import time
from bs4 import BeautifulSoup
from pathlib import Path
from werkzeug.utils import secure_filename
import shutil
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode
# from io import BytesIO
# from docling.datamodel.base_models import DocumentStream
from app.convert import docling_to_custom_json
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_RESOLUTION_SCALE = 2.0
CONFIG_FILE = "config.json"

def loadConfig():
    """
    Load configuration from the config file or return default configuration.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            logger.info("Loaded configuration file.")
            return json.load(f)
    logger.warning("Configuration file not found. Using defaults.")
    return {
        "output_directory": r"data/IngestedFiles",
        "temp_directory": r"data/TempFiles",
        "mapping_file": r"client_mapping.json",
        "supported_formats": ["pdf", "docx", "xlsx", "odt", "ods", "png", "tiff", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel", ".pdf"],
        "template_folder": "Docling\project\templates",
        "static_folder": "Docling\project\static"
    }

config = loadConfig()

def loadClientMapping():
    """
    Load client mapping from the mapping file or return an empty mapping.
    """
    mapping_file = config["mapping_file"]
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            logger.info("Loaded client mapping.")
            return json.load(f), mapping_file
    logger.warning("Client mapping file not found. Starting fresh.")
    return {}, mapping_file

def saveClientMapping(client_mapping, mapping_file):
    """
    Save client mapping to the mapping file.
    """
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)
        logger.info("Saved client mapping.")

def generateClientID(client_mapping):
    """
    Generate a unique 8-digit Client ID.
    """
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            logger.info(f"Generated new Client ID: {client_id}")
            return client_id

def getClientOutputDir(client_id):
    """
    Get the output directory for a given Client ID.
    """
    output_dir = Path(config["output_directory"]) / client_id
    logger.info(f"Resolved output directory for Client ID {client_id}: {output_dir}")
    return output_dir

def processDocument(file_path, output_dir, global_client_id):
    """
    Process the document using Docling and extract figures and tables.
    """
    logger.info(f"Starting document processing for {file_path}.")
    try:
        start_time = time.time()

        # Configure pipeline options for figures and tables
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
        pipeline_options.generate_picture_images = True
        pipeline_options.generate_table_images = True

        # Set up the document converter
        doc_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        # Convert the document
        conv_result = doc_converter.convert(file_path)

        # Check conversion status
        if conv_result.status != ConversionStatus.SUCCESS:
            end_time = time.time()
            logger.info(f"Processing time: {end_time - start_time:.2f} seconds; It ended in a Failure")
            logger.error(f"Failed to process {file_path}. Status: {conv_result.status}")
            return {"error": f"Failed to process {file_path}. Status: {conv_result.status}"}

        # Process figures and tables
        figure_counter = 0
        table_counter = 0
        for element, _ in conv_result.document.iterate_items():
            if isinstance(element, PictureItem):  # Save figures as PNG
                figure_counter += 1
                figure_path = output_dir / f"{global_client_id}-figure-{figure_counter}.png"
                with figure_path.open("wb") as fp:
                    element.get_image(conv_result.document).save(fp, "PNG")
            elif isinstance(element, TableItem):  # Save tables as PNG and HTML
                table_counter += 1
                table_html_path = output_dir / f"{global_client_id}-table-{table_counter}.html"
                with table_html_path.open("w", encoding="utf-8") as fp:
                    fp.write(element.export_to_html())

        # Save document as HTML with image references
        json_filename = output_dir / f"{global_client_id}-with-image-refs.html"
        conv_result.document.save_as_html(json_filename, image_mode=ImageRefMode.REFERENCED)

        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds. Successfully processed {file_path}")

        # Convert extracted HTML to JSON
        json_conversion_result = docling_to_custom_json(global_client_id, output_dir)
        if "error" in json_conversion_result:
            logger.error(f"JSON conversion failed: {json_conversion_result['error']}")
            return {"error": json_conversion_result["error"]}

        # Delete artifacts folder after JSON conversion
        artifacts_folder = output_dir / f"{global_client_id}-with-image-refs_artifacts"
        if artifacts_folder.exists():
            shutil.rmtree(artifacts_folder)
            logger.info(f"Deleted artifacts folder: {artifacts_folder}")

        return {
            "message": (
                f"File processed successfully with {figure_counter} figures and "
                f"{table_counter} tables saved."
            ),
            "output_dir": str(output_dir),
            "output_json": json_conversion_result["output_path"]
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        return {"error": str(e)}
   