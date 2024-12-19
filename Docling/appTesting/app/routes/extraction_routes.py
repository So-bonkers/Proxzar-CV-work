from flask import Blueprint, request, jsonify
from services.document_processor import process_document
from pathlib import Path

extraction_blueprint = Blueprint('extraction', __name__)

@extraction_blueprint.route('/api/v1/extract', methods=['POST'])
def extract():
    client_id = request.form.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID is required"}), 400

    # Assume `client_mapping` is loaded here
    client_mapping = None  # Placeholder for actual client mapping loading logic
    if not client_mapping:
        return jsonify({"error": "Client mapping not found"}), 400

    client_data = {"output_path": "output_dir", "saved_file": "file_path"}
    saved_file_path = Path(client_data["saved_file"])
    output_path = Path(client_data["output_path"])

    if not saved_file_path.exists():
        return jsonify({"error": f"Original file not found: {saved_file_path}"}), 404

    result = process_document(saved_file_path, output_path, global_client_id=client_id)
    if "error" in result:
        return jsonify({"error": result["error"]}), 500

    return jsonify({"message": "Document extracted successfully", "output_html": result["output_html"]})
