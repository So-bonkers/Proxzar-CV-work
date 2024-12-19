from flask import Blueprint, request, jsonify, render_template
from services.utils import generate_client_id, validate_file_format, save_temp_file, move_file_to_output
from config.config_loader import load_config
import json
from pathlib import Path

ingestion_blueprint = Blueprint('ingestion', __name__)
config = load_config("config.json")

@ingestion_blueprint.route('/api/v1/ingest', methods=['POST'])
# Routes
def index():
    """Serve the HTML UI."""
    return render_template(r'templates\index.html')

def ingest():
    """
    Handle file ingestion.
    - Validate the file format.
    - Generate a unique client ID.
    - Save the file to a temporary directory.
    - Move the file to the output directory.
    """
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    if not validate_file_format(file.filename, config["supported_formats"]):
        return jsonify({"error": f"Unsupported file format: {file.filename}"}), 400

    client_id = generate_client_id(client_mapping={})
    output_path = Path(config["output_directory"]) / client_id
    output_path.mkdir(parents=True, exist_ok=True)

    saved_file_path = save_temp_file(file, config["temp_directory"])
    output_file_path = move_file_to_output(saved_file_path, output_path)

    return jsonify({"client_id": client_id, "output_path": str(output_file_path)})
