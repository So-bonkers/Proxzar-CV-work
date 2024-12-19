from flask import Blueprint, jsonify, send_from_directory
from pathlib import Path
import logging

# Create a Blueprint for download-related routes
download_bp = Blueprint('download_routes', __name__)
logger = logging.getLogger(__name__)

def get_client_output_dir(client_id, base_dir="data/IngestedFiles"):
    """
    Get the output directory for a given Client ID.
    """
    return Path(base_dir) / client_id

@download_bp.route('/download/<client_id>/<filename>', methods=['GET'])
def download_file(client_id, filename):
    """
    Serve files for download based on client ID and file name.
    """
    try:
        output_dir = get_client_output_dir(client_id)
        file_path = output_dir / filename

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return jsonify({"error": f"File not found: {filename}"}), 404

        return send_from_directory(output_dir, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename} for client {client_id}: {e}")
        return jsonify({"error": f"Unable to serve file: {e}"}), 500
