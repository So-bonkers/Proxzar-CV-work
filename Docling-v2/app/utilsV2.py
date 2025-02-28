import os
import random
# import requests
from urllib.parse import urlparse
import json
import logging
import time
# from bs4 import BeautifulSoup
from pathlib import Path
from werkzeug.utils import secure_filename
import shutil
import pandas as pd
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode
# from io import BytesIO
# from docling.datamodel.base_models import DocumentStream
from app.convertV2 import docling_to_custom_json
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

def commonProcessing(conv_result, output_dir: Path, file_name: str):
    """
    Performs the common processing tasks for documents including OCR, table extraction,
    figure extraction, and exporting to various formats.

    Args:
        conv_result: The conversion result from Docling.
        output_dir (Path): The directory where processed files will be stored.
        file_name (str): The name of the file being processed.

    Returns:
        dict: A dictionary containing processing details.
    """
    if conv_result.status != ConversionStatus.SUCCESS:
        logger.error(f"Conversion failed for {file_name}. Status: {conv_result.status}")
        return {"error": f"Processing failed. Status: {conv_result.status}"}

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save tables as HTML
    table_counter = 0
    for table in conv_result.document.tables:
        table_df: pd.DataFrame = table.export_to_dataframe()
        table_counter += 1
        table_html_path = output_dir / f"{file_name}-table-{table_counter}.html"
        with table_html_path.open("w", encoding="utf-8") as fp:
            fp.write(table.export_to_html())

    # Save page images
    for page_no, page in conv_result.document.pages.items():
        if page.image is not None:
            page_image_path = output_dir / f"{file_name}-page-{page_no}.png"
            with page_image_path.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

    # Save extracted figures and tables as images
    figure_counter = 0
    table_counter = 0
    for element, _ in conv_result.document.iterate_items():
        if isinstance(element, PictureItem):
            figure_counter += 1
            element_image_path = output_dir / f"{file_name}-figure-{figure_counter}.png"
        elif isinstance(element, TableItem):
            table_counter += 1
            element_image_path = output_dir / f"{file_name}-table-{table_counter}.png"
        else:
            continue

        with element_image_path.open("wb") as fp:
            if element.get_image(conv_result.document) is not None:
                element.get_image(conv_result.document).save(fp, "PNG")

    # Save Markdown with internal and external references
    md_internal = output_dir / f"{file_name}-with-images-internal.md"
    conv_result.document.save_as_markdown(md_internal, image_mode=ImageRefMode.EMBEDDED)

    md_external = output_dir / f"{file_name}-with-images-external-refs.md"
    conv_result.document.save_as_markdown(md_external, image_mode=ImageRefMode.REFERENCED)

    # Save HTML
    html_filename = output_dir / f"{file_name}-with-images.html"
    conv_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    html_no_images = output_dir / f"{file_name}-without-images.html"
    with html_no_images.open("w", encoding="utf-8") as fp:
        fp.write(conv_result.document.export_to_html())

    json_conversion_result = docling_to_custom_json(file_name, output_dir)
    if "error" in json_conversion_result:
        logger.error(f"JSON conversion failed: {json_conversion_result['error']}")
        return {"error": json_conversion_result["error"]}

    return {
        "message": f"File {file_name} processed successfully.",
        "output_dir": str(output_dir),
        "output_json": json_conversion_result["output_path"]
    }

def processDocument(file_path: str, output_dir: Path, file_name: str):
    """
    Processes a document uploaded by the user, extracting text, figures, tables, and images.

    Args:
        file_path (str): Path to the uploaded file.
        output_dir (Path): Directory for storing output.
        file_name (str): Name of the file being processed.

    Returns:
        dict: Processing result including extracted text, figures, and tables.
    """
    logger.info(f"Starting document processing for: {file_name}")

    start_time = time.time()

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR
    pipeline_options.ocr_options.use_gpu = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    conv_result = doc_converter.convert(file_path)

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds for {file_name}")

    return commonProcessing(conv_result, output_dir, file_name)

import boto3
from io import BytesIO
from docling.datamodel.base_models import DocumentStream

def processStreamDocument(bucket_name: str, file_key: str, output_dir: Path, file_name: str):
    """
    Processes a document from an S3 stream, extracting text, figures, tables, and images.

    Args:
        bucket_name (str): S3 bucket name.
        file_key (str): Key of the file in S3.
        output_dir (Path): Directory for storing output.
        file_name (str): Name of the file being processed.

    Returns:
        dict: Processing result including extracted text, figures, and tables.
    """
    logger.info(f"Starting stream processing for S3 file: {file_key} from bucket {bucket_name}")

    start_time = time.time()
    s3_client = boto3.client("s3")

    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    binary_stream = response["Body"].read()

    buf = BytesIO(binary_stream)
    source = DocumentStream(name=file_key, stream=buf)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True  # Enable OCR
    pipeline_options.ocr_options.use_gpu = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    conv_result = doc_converter.convert(source)

    end_time = time.time()
    logger.info(f"Processing completed in {end_time - start_time:.2f} seconds for {file_key}")

    return commonProcessing(conv_result, output_dir, file_name)
    