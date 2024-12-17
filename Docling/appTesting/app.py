from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import os
import random
import json
import logging
import time
from werkzeug.utils import secure_filename
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import FigureElement, InputFormat, Table
from docling_core.types.doc import PictureItem, TableItem

IMAGE_RESOLUTION_SCALE = 2.0  # Scale for image resolution

# Flask setup
app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_FILE = "config.json"
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE) as f:
        config = json.load(f)
else:
    config = {
        "output_directory": r"data\IngestedFiles",
        "temp_directory": r"data\TempFiles",
        "mapping_file": r"client_mapping.json",
        "supported_formats": [".pdf", ".docx", ".xlsx", ".odt", ".ods", ".png", ".tiff"]
    }

# Ensure required directories exist
Path(config["temp_directory"]).mkdir(parents=True, exist_ok=True)
Path(config["output_directory"]).mkdir(parents=True, exist_ok=True)

# Load or initialize the client mapping
mapping_file = config["mapping_file"]
if os.path.exists(mapping_file):
    with open(mapping_file) as f:
        client_mapping = json.load(f)
else:
    client_mapping = {}

# Helper Functions
def generate_client_id():
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

def process_document(file_path, output_dir):
    """
    Process the document using Docling, extracting text, figures, and tables.
    Save extracted content as JSON, figures as PNG, and tables as both HTML and PNG.
    References for figures and tables are included in the JSON file.
    """
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
        json_output = {"text": [], "figures": [], "tables": []}  # Initialize JSON structure

        # Process elements (figures and tables)
        figure_counter = 0
        table_counter = 0

        for element, _ in conv_result.document.iterate_items():
            if isinstance(element, PictureItem):  # Save figures as PNG
                figure_counter += 1
                figure_path = output_dir / f"{doc_filename}-figure-{figure_counter}.png"
                with figure_path.open("wb") as fp:
                    element.get_image(conv_result.document).save(fp, "PNG")
                # Add figure reference to JSON
                json_output["figures"].append({"id": figure_counter, "path": str(figure_path.name)})

            elif isinstance(element, TableItem):  # Save tables as PNG and HTML
                table_counter += 1

                # Save table as PNG
                table_image_path = output_dir / f"{doc_filename}-table-{table_counter}.png"
                with table_image_path.open("wb") as fp:
                    element.get_image(conv_result.document).save(fp, "PNG")

                # Save table as standalone HTML
                table_html_path = output_dir / f"{doc_filename}-table-{table_counter}.html"
                with table_html_path.open("w", encoding="utf-8") as fp:
                    fp.write(element.table.to_html(index=False, escape=False))  # Convert table to HTML

                # Add table reference to JSON
                json_output["tables"].append({
                    "id": table_counter,
                    "html_path": str(table_html_path.name),
                    "image_path": str(table_image_path.name)
                })

        # Add textual content to JSON
        for page in conv_result.document.pages.values():
            json_output["text"].append(page.text)

        # Save the JSON output
        json_path = output_dir / f"{doc_filename}.json"
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(json_output, fp, indent=4)

        end_time = time.time()
        logger.info(f"Processing time: {end_time - start_time:.2f} seconds. Successfully processed {file_path}")
        return {
            "message": (
                f"File processed successfully with {figure_counter} figures and "
                f"{table_counter} tables saved."
            ),
            "output_dir": str(output_dir),
            "output_json": str(json_path),
        }

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return {"error": str(e)}

# Routes
@app.route('/')
def index():
    """Serve the HTML UI."""
    return render_template('index.html')

@app.route('/api/v1/ingest', methods=['POST'])
def ingest():
    """Handle file ingestion."""
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if not validate_file_format(file.filename):
        return jsonify({"error": f"Unsupported file format: {file.filename}"}), 400

    # Assign Client ID
    client_id = generate_client_id()
    output_path = get_client_output_dir(client_id)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the file in the ClientID directory
    saved_file_path = output_path / secure_filename(file.filename)
    file.save(saved_file_path)

    # Update client mapping
    client_mapping[client_id] = {
        "original_file": file.filename,
        "output_path": str(output_path),
        "saved_file": str(saved_file_path)
    }
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)

    return jsonify({"client_id": client_id, "output_path": str(output_path)})

@app.route('/api/v1/extract', methods=['POST'])
def extract():
    """Handle document extraction."""
    client_id = request.form.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID is required"}), 400

    if client_id not in client_mapping:
        return jsonify({"error": f"Client ID {client_id} not found"}), 404

    # Retrieve file and output path
    client_data = client_mapping[client_id]
    saved_file_path = Path(client_data["saved_file"])
    output_path = Path(client_data["output_path"])

    if not saved_file_path.exists():
        return jsonify({"error": f"Original file not found: {saved_file_path}"}), 404

    # Process the document
    result = process_document(saved_file_path, output_path)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({
        "message": "Document extracted successfully",
        "output_directory": result["output_dir"]
    })

@app.route('/download/<client_id>/<filename>')
def download_file(client_id, filename):
    """Serve files for download."""
    output_dir = get_client_output_dir(client_id)
    return send_from_directory(output_dir, filename)

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
