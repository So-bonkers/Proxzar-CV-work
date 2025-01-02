import os
import random
import requests
from urllib.parse import urlparse
import json
import logging
import time
from bs4 import BeautifulSoup
from pathlib import Path
from werkzeug.utils import secure_filename
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode
import boto3
from io import BytesIO
from docling.datamodel.base_models import DocumentStream

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
IMAGE_RESOLUTION_SCALE = 2.0
CONFIG_FILE = "config.json"

def loadConfig():
    """
    Load configuration from the config file or return default configuration.
    
    Returns:
        dict: Configuration dictionary.
    """
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            logger.info("Loaded configuration file.")
            return json.load(f)
    logger.warning("Configuration file not found. Using defaults.")
    return {
        "output_directory": r"data\IngestedFiles",
        "temp_directory": r"data\TempFiles",
        "mapping_file": r"client_mapping.json",
        "supported_formats": ["pdf", "docx", "xlsx", "odt", "ods", "png", "tiff", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel"],
        "template_folder": "Docling\\project\\templates",
        "static_folder": "Docling\\project\\static"
    }

config = loadConfig()

def loadClientMapping():
    """
    Load client mapping from the mapping file or return an empty mapping.
    
    Returns:
        tuple: A tuple containing the client mapping dictionary and the mapping file path.
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
    
    Args:
        client_mapping (dict): The client mapping dictionary.
        mapping_file (str): The path to the mapping file.
    """
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)
        logger.info("Saved client mapping.")

def generateClientID(client_mapping):
    """
    Generate a unique 8-digit Client ID.
    
    Args:
        client_mapping (dict): The client mapping dictionary.
    
    Returns:
        str: A unique 8-digit Client ID.
    """
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            logger.info(f"Generated new Client ID: {client_id}")
            return client_id

def validateFileFormat(filename_or_url):
    """
    Check if the file or URL has a supported format.
    
    Args:
        filename_or_url (str): The filename or URL to validate.
    
    Returns:
        bool: True if the format is supported, False otherwise.
    """
    _, ext = os.path.splitext(filename_or_url.lower())
    
    if filename_or_url.startswith("https://arxiv.org/pdf/"):
        logger.info(f"Assuming PDF format for arXiv link: {filename_or_url}")
        return True
    
    # Check by extension if available
    if ext in config["supported_formats"]:
        logger.info(f"File format validation for {filename_or_url}: valid (by extension)")
        return True

    # Handle URL validation
    parsed_url = urlparse(filename_or_url)
    if parsed_url.scheme in ["http", "https"]:
        try:
            response = requests.head(filename_or_url, allow_redirects=True)
            content_type = response.headers.get("Content-Type", "").lower()
            logger.info(f"Content-Type for URL {filename_or_url}: {content_type}")

            # Match MIME types for known formats
            mime_types = {
                ".pdf": "application/pdf",
                ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".png": "image/png",
                ".tiff": "image/tiff",
            }
            if any(mime in content_type for mime in mime_types.values()):
                logger.info(f"File format validation for {filename_or_url}: valid (by MIME type)")
                return True
        except requests.RequestException as e:
            logger.error(f"Error validating file format from URL: {e}")
            return False

    logger.info(f"File format validation for {filename_or_url}: invalid")
    return False

def getClientOutputDir(client_id):
    """
    Get the output directory for a given Client ID.
    
    Args:
        client_id (str): The Client ID.
    
    Returns:
        Path: The output directory path.
    """
    output_dir = Path(config["output_directory"]) / client_id
    logger.info(f"Resolved output directory for Client ID {client_id}: {output_dir}")
    return output_dir

def processDocument(file_path, output_dir, global_client_id):
    """
    Process the document to extract content grouped by pages with external references for images and tables.
    
    Args:
        file_path (str): Path to the input file.
        output_dir (Path): Output directory for processed files.
        global_client_id (str): Unique identifier for the client.
    
    Returns:
        dict: Information about processing results.
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
            logger.error(f"Failed to process {file_path}. Status: {conv_result.status}")
            return {"error": f"Failed to process {file_path}. Status: {conv_result.status}"}

        # Build HTML with grouped content by pages
        html_content = []
        figure_counter = 0
        table_counter = 0

        for page_number, page in enumerate(conv_result.document.pages.values(), start=1):
            page_content = f'<div class="page" data-page-number="{page_number}">'
            page_content += f'<p>{page.text.strip()}</p>'  # Add text content
            
            for element in page.iterate_items():
                if isinstance(element, PictureItem):
                    figure_counter += 1
                    figure_path = output_dir / f"{global_client_id}-figure-{figure_counter}.png"
                    element.get_image(conv_result.document).save(figure_path, "PNG")
                    page_content += f'<img src="data/IngestedFiles/{global_client_id}/{global_client_id}-figure-{figure_counter}.png" />'
                elif isinstance(element, TableItem):
                    table_counter += 1
                    table_html_path = output_dir / f"{global_client_id}-table-{table_counter}.html"
                    with table_html_path.open("w", encoding="utf-8") as fp:
                        fp.write(element.export_to_html())
                    page_content += f'<table data-ref="data/IngestedFiles/{global_client_id}/{global_client_id}-table-{table_counter}.html"></table>'
            
            page_content += '</div>'
            html_content.append(page_content)

        # Save the document as HTML with page grouping
        html_filename = output_dir / f"{global_client_id}-with-image-refs.html"
        with open(html_filename, "w", encoding="utf-8") as html_file:
            html_file.write("<html><body>" + "".join(html_content) + "</body></html>")

        end_time = time.time()
        logger.info(f"Successfully processed {file_path} in {end_time - start_time:.2f} seconds.")
        return {
            "message": (
                f"File processed successfully with {figure_counter} figures and {table_counter} tables."
            ),
            "output_html": str(html_filename),
        }
    except Exception as e:
        logger.error(f"Error processing document {file_path}: {e}")
        raise


def getS3Client():
    """
    Initialize and return an S3 client using configuration.
    
    Returns:
        boto3.client: The S3 client.
    """
    try:
        with open("aws_access_config.json") as f:
            config = json.load(f)
        
        return boto3.client(
            's3',
            aws_access_key_id=config['Access-key'],
            aws_secret_access_key=config['Secret-key'],
            region_name='us-east-2'
        )
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        raise

def processStreamDocument(bucket_name, file_key, output_dir, client_id):
    """
    Process a document directly from S3 stream.
    
    Args:
        bucket_name (str): The name of the S3 bucket.
        file_key (str): The key of the file in the S3 bucket.
        output_dir (Path): The output directory path.
        client_id (str): The Client ID.
    
    Returns:
        dict: A dictionary containing the processing result.
    """
    logger.info(f"Starting stream document processing for {file_key} from bucket {bucket_name}.")
    try:
        start_time = time.time()

        # Initialize S3 client
        s3_client = getS3Client()

        # Stream the file from S3
        response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        binary_stream = response['Body'].read()

        # Create document stream
        buf = BytesIO(binary_stream)
        source = DocumentStream(name=file_key, stream=buf)

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
        conv_result = doc_converter.convert(source)

        # Check conversion status
        if conv_result.status != ConversionStatus.SUCCESS:
            end_time = time.time()
            logger.info(f"Processing time: {end_time - start_time:.2f} seconds; It ended in a Failure")
            logger.error(f"Failed to process stream from {bucket_name}/{file_key}. Status: {conv_result.status}")
            return {"error": f"Failed to process stream. Status: {conv_result.status}"}

        # Process elements (figures and tables)
        figure_counter = 0
        table_counter = 0
        for element, _ in conv_result.document.iterate_items():
            if isinstance(element, PictureItem):
                figure_counter += 1
                figure_path = output_dir / f"{client_id}-figure-{figure_counter}.png"
                with figure_path.open("wb") as fp:
                    element.get_image(conv_result.document).save(fp, "PNG")
            elif isinstance(element, TableItem):
                table_counter += 1
                table_html_path = output_dir / f"{client_id}-table-{table_counter}.html"
                with table_html_path.open("w", encoding="utf-8") as fp:
                    fp.write(element.export_to_html())

        # Save the document as HTML with referenced figures and tables
        html_filename = output_dir / f"{client_id}-with-image-refs.html"
        conv_result.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds. Successfully processed stream from {bucket_name}/{file_key}")
        return {
            "message": f"Stream processed successfully with {figure_counter} figures and {table_counter} tables saved.",
            "output_dir": str(output_dir),
            "output_html": str(html_filename)
        }

    except Exception as e:
        logger.error(f"Error processing stream document from {bucket_name}/{file_key}: {e}")
        raise

def getContentType(element):
    """
    Determine the content type of an HTML element and extract its data.
    
    Args:
        element (Tag): A BeautifulSoup Tag object representing an HTML element.
    
    Returns:
        dict: A dictionary containing the content kind and its source data.
    """
    if element.name == 'p':
        return {"contentType": "paragraph", "source": element.text.strip()}
    elif element.name in ['ul', 'ol']:
        list_items = [li.text.strip() for li in element.find_all('li')]
        return {"contentType": "list", "source": list_items}
    elif element.name == 'table':
        rows = []
        for tr in element.find_all('tr'):
            row = []
            for cell in tr.find_all(['th', 'td']):
                row.append(cell.text.strip())
            rows.append(row)
        return {"contentType": "table", "source": rows}
    elif element.name == 'img':
        return {"contentType": "image", "source": element['src']}
    return {}

def parseHTMLToJSON(html_file_path, json_file_path):
    """
    Parse an HTML document with external references, grouped by pages, and convert it to JSON.
    
    Args:
        html_file_path (str): Path to the HTML file.
        json_file_path (str): Path to save the JSON output.
    """
    try:
        with open(html_file_path, "r", encoding="utf-8") as html_file:
            soup = BeautifulSoup(html_file, "html.parser")

        json_output = {"pages": []}

        # Extract content page by page
        for page_div in soup.find_all("div", class_="page"):
            page_number = int(page_div.get("data-page-number", 0))
            page_data = {"page_number": page_number, "content": []}

            for element in page_div.children:
                if element.name == "img":
                    page_data["content"].append({
                        "type": "image",
                        "path": element["src"],
                    })
                elif element.name == "table":
                    table_path = element.get("data-ref", None)
                    if table_path:
                        page_data["content"].append({
                            "type": "table",
                            "path": table_path,
                        })
                elif element.name in ["p", "h1", "h2", "h3", "h4", "h5", "h6"]:
                    page_data["content"].append({
                        "type": "text",
                        "content": element.get_text(strip=True),
                    })

            json_output["pages"].append(page_data)

        # Save JSON output
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump(json_output, json_file, indent=4)

        logger.info(f"Converted HTML to JSON and saved to {json_file_path}.")
    except Exception as e:
        logger.error(f"Error parsing HTML to JSON: {e}")
        raise
