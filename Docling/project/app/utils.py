import os
import random
import json
import logging
import time
from pathlib import Path
from werkzeug.utils import secure_filename
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGE_RESOLUTION_SCALE = 2.0

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

        # Prepare for saving outputs
        doc_filename = file_path.stem
        # json_output = {"text": [], "figures": [], "tables": []}  # Initialize JSON structure

        # Process elements (figures and tables)
        figure_counter = 0
        table_counter = 0
        for element, _ in conv_result.document.iterate_items():
            if isinstance(element, PictureItem):  # Save figures as PNG
                figure_counter += 1
                figure_path = output_dir / f"{global_client_id}-figure-{figure_counter}.png"
                with figure_path.open("wb") as fp:
                    element.get_image(conv_result.document).save(fp, "PNG")
                # Add figure reference to JSON
                # json_output["figures"].append({"id": figure_counter, "path": str(figure_path.name)})
            elif isinstance(element, TableItem):  # Save tables as PNG and HTML
                table_counter += 1
                # Save table as PNG
                # table_image_path = output_dir / f"{doc_filename}-table-{table_counter}.png"
                # with table_image_path.open("wb") as fp:
                #     element.get_image(conv_result.document).save(fp, "PNG")
                # Save table as standalone HTML
                table_html_path = output_dir / f"{global_client_id}-table-{table_counter}.html"
                with table_html_path.open("w", encoding="utf-8") as fp:
                    print("Exporting to HTML: command is executing now")
                    fp.write(element.export_to_html())


                # # Add table reference to JSON
                # json_output["tables"].append({
                #     "id": table_counter,
                #     "html_path": str(table_html_path.name),
                #     "image_path": str(table_image_path.name)
                # })
        
        # # Add textual content to JSON
        # for page in conv_result.document.pages.values():
        #     json_output["text"].append(page.text)

        # # Save the JSON output
        # json_path = output_dir / f"{doc_filename}.json"
        # with json_path.open("w", encoding="utf-8") as fp:
        #     json.dump(json_output, fp, indent=4)

        # Save the document as HTML with referenced figures and tables
        html_filename = output_dir / f"{global_client_id}-with-image-refs.html"
        print("Exporting to HTML, main doc: command is executing now")
        conv_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds. Successfully processed {file_path}")
        return {
            "message": (
                f"File processed successfully with {figure_counter} figures and "
                f"{table_counter} tables saved."
            ),
            "output_dir": str(output_dir),
            # "output_json": str(json_path),
            "output_html": str(html_filename)
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise
