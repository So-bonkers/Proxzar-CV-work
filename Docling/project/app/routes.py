import logging
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from app.utils import (
    generate_client_id,
    validate_file_format,
    get_client_output_dir,
    process_document,
    load_client_mapping,
    save_client_mapping,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_blueprint = Blueprint("main", __name__)
client_mapping, mapping_file = load_client_mapping()

@main_blueprint.route('/')
def index():
    """Serve the HTML UI."""
    logger.info("Rendering index page.")
    return render_template('index.html')

@main_blueprint.route('/api/v1/ingest', methods=['POST'])
def ingest():
    """Handle file ingestion."""
    file = request.files.get('file')
    if not file:
        logger.error("No file provided for ingestion.")
        return jsonify({"error": "No file provided"}), 400

    if not validate_file_format(file.filename):
        logger.error(f"Unsupported file format: {file.filename}")
        return jsonify({"error": f"Unsupported file format: {file.filename}"}), 400

    client_id = generate_client_id(client_mapping)
    output_path = get_client_output_dir(client_id)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_file_path = output_path / secure_filename(file.filename)
    file.save(saved_file_path)

    client_mapping[client_id] = {
        "original_file": file.filename,
        "output_path": str(output_path),
        "saved_file": str(saved_file_path),
    }
    save_client_mapping(client_mapping, mapping_file)

    logger.info(f"File {file.filename} ingested successfully with Client ID {client_id}.")
    return jsonify({"client_id": client_id, "output_path": str(output_path)})

@main_blueprint.route('/api/v1/extract', methods=['POST'])
def extract():
    """Handle document extraction."""
    client_id = request.form.get('client_id')
    if not client_id:
        logger.error("Client ID is missing from extraction request.")
        return jsonify({"error": "Client ID is required"}), 400

    if client_id not in client_mapping:
        logger.error(f"Client ID {client_id} not found in mapping.")
        return jsonify({"error": f"Client ID {client_id} not found"}), 404

    client_data = client_mapping[client_id]
    saved_file_path = Path(client_data["saved_file"])
    output_path = Path(client_data["output_path"])

    if not saved_file_path.exists():
        logger.error(f"Original file not found: {saved_file_path}")
        return jsonify({"error": f"Original file not found: {saved_file_path}"}), 404

    result = process_document(saved_file_path, output_path, global_client_id=client_id)
    if "error" in result:
        logger.error(f"Error during extraction: {result['error']}")
        return jsonify({"error": result["error"]}), 500

    logger.info(f"Document extracted successfully for Client ID {client_id}.")
    return jsonify({
        "message": "Document extracted successfully",
        "output_directory": result["output_dir"]
    })

@main_blueprint.route('/download/<client_id>/<filename>')
def download_file(client_id, filename):
    """Serve files for download."""
    logger.info(f"Serving download request for Client ID {client_id}, file {filename}.")
    output_dir = get_client_output_dir(client_id)
    return send_from_directory(output_dir, filename)
