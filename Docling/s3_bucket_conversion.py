import logging
import json
from pathlib import Path
from urllib.parse import urlparse
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

# Define supported formats
SUPPORTED_FORMATS = {'pdf', 'docx', 'xlsx', 'odt', 'ods', 'png', 'jpg', 'jpeg', 'tiff', 'pptx', 'html', 'asciidoc', 'md'}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_and_save_locally(input_path, local_output_path):
    """Convert a file to Docling format and save it as JSON to a local path."""
    logger.info(f"Converting {input_path} to Docling format.")
    
    # Set up pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    # Configure the Docling document converter
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options, backend=PyPdfiumDocumentBackend
            )
        }
    )

    # Convert the document
    doc = doc_converter.convert(Path(input_path))

    # Serialize to JSON string
    doc_json = json.dumps(doc.dict(), indent=4)

    # Save JSON to the local path
    output_file = Path(local_output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directories exist
    with open(output_file, 'w') as f:
        f.write(doc_json)
    
    logger.info(f"Converted file saved locally at {output_file}")

def process_file(link, local_output_dir):
    """Process a single file link and save locally."""
    parsed = urlparse(link)
    file_path = Path(parsed.path)
    file_extension = file_path.suffix.lower().strip('.')

    if file_extension not in SUPPORTED_FORMATS:
        logger.error(f"Unsupported file format: {file_extension}")
        return

    # Determine local output path
    output_file_name = file_path.stem + ".json"
    local_output_path = Path(local_output_dir) / output_file_name

    # Convert and save locally
    convert_and_save_locally(file_path, local_output_path)
    logger.info(f"Processed file saved to {local_output_path}")

# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python convert_to_docling.py <input_file_link_or_path> <local_output_dir>")
        sys.exit(1)

    input_link = sys.argv[1]
    local_output_dir = sys.argv[2]

    logger.info(f"Processing file: {input_link}")
    process_file(input_link, local_output_dir)
