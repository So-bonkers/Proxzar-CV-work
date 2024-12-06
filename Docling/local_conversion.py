from pathlib import Path
import mimetypes
import re
import json
import tempfile
import requests
import logging
from urllib.parse import urlparse
from io import BytesIO
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Define supported formats
SUPPORTED_FORMATS = {'pdf', 'docx', 'xlsx', 'odt', 'ods', 'png', 'jpg', 'jpeg', 'tiff', 'pptx', 'html', 'asciidoc', 'md'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def determine_file_format(link):
    """Determine the file format from the URL."""
    # Define regex patterns for supported file formats (without the dot)
    file_extension_patterns = [
        r"pdf", r"docx", r"xlsx", r"odt", r"ods", 
        r"png", r"jpg", r"jpeg", r"tiff", r"pptx", 
        r"html", r"asciidoc", r"md"
    ]
    
    # First, check if the URL ends with a valid file extension
    for pattern in file_extension_patterns:
        if re.search(pattern, link, re.IGNORECASE):
            print("Allowed extension found at the end: ", pattern)
            return pattern  # Return the file extension (without dot)
    
    # If no valid extension is found at the end, check the section before the last "/"
    path_before_last_slash = link.rsplit('/', 1)[0]  # Get the part of the link before the last '/'
    
    # Now check this section before the last '/'
    for pattern in file_extension_patterns:
        if re.search(pattern, path_before_last_slash, re.IGNORECASE):
            print("Allowed extension found before the last slash: ", pattern)
            return pattern  # Return the file extension (without dot)
    
    # If no match found, log an error and return None
    logger.error(f"Unsupported file format for link: {link}")
    return None

def convert_from_stream(stream, file_extension, output_path):
    """Convert the streamed file data to Docling format."""
    
    try:
        # If the stream is in bytes, create a file from it
        if isinstance(stream, bytes):
            # Write bytes to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
                temp_file.write(stream)  # Write the byte content to the file
                temp_file_path = temp_file.name  # Get the file path

        else:
            # If it's not in bytes, it's probably a file-like object (like BytesIO)
            with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
                temp_file.write(stream.read())  # Write the content of the stream to the file
                temp_file_path = temp_file.name  # Get the file path

        # Now pass the temporary file path to the converter
        logger.info(f"Converting file from temporary location: {temp_file_path}")
        doc = DocumentConverter.convert(temp_file_path, additional_param="value")  # Assuming convert() expects a path

        # Process the conversion as needed
        logger.info(f"Conversion successful for {temp_file_path}. Saving result to {output_path}")
        # Perform any post-processing or saving operations here

    except Exception as e:
        logger.error(f"Error during conversion: {e}")
def process_file(link, local_output_dir):
    """Process a single file link and convert directly from stream."""
    file_extension = determine_file_format(link)

    if not file_extension or file_extension not in SUPPORTED_FORMATS:
        logger.error(f"Unsupported file format: {file_extension}")
        return

    logger.info(f"Detected file format: {file_extension}")

    # Fetch the file as a stream
    logger.info(f"Fetching the file from {link}")
    response = requests.get(link, stream=True)
    response.raise_for_status()

    # Read content into memory
    stream = response.content

    # Determine local output path
    output_file_name = Path(link).stem + ".json"
    local_output_path = Path(local_output_dir) / output_file_name

    # Convert and save locally
    convert_from_stream(stream, file_extension, local_output_path)

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python convert_to_docling.py <input_file_link> <local_output_dir>")
        sys.exit(1)

    input_link = sys.argv[1]
    local_output_dir = sys.argv[2]

    logger.info(f"Processing file: {input_link}")
    process_file(input_link, local_output_dir)
