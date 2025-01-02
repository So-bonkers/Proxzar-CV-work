# routes.py
import logging
from flask import Blueprint, request, jsonify, render_template
from app.services.ingestion_service import handle_file_ingestion
from app.utils import loadClientMapping
from app.services.document_processing_service import handle_document_processing
from app.services.link_processing_service import handle_link_ingestion
from app.services.stream_ingestion_service import handle_stream_ingestion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

main_blueprint = Blueprint("main", __name__)

# Load client mapping
client_mapping, mapping_file = loadClientMapping()

@main_blueprint.route('/')
def index():
    return render_template('index.html')

@main_blueprint.route('/api/v1/ingest', methods=['POST'])
def ingest():
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file provided"}), 400

    try:
        result = handle_file_ingestion(file, client_mapping, mapping_file)
        return jsonify(result), 200
    except ValueError as e:
        logger.error(f"Ingestion error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@main_blueprint.route('/api/v1/process', methods=['POST'])
def process_document():
    data = request.get_json()
    client_id = data.get('client_id')
    if not client_id:
        return jsonify({"error": "Client ID is required"}), 400

    try:
        result = handle_document_processing(client_id, client_mapping)
        return jsonify(result), 200
    except ValueError as e:
        logger.error(f"Processing error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during processing: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@main_blueprint.route('/api/v1/ingest-link', methods=['POST'])
def ingest_link():
    data = request.get_json()
    file_link = data.get('file_link')
    if not file_link:
        return jsonify({"error": "File link is required"}), 400

    try:
        result = handle_link_ingestion(file_link, client_mapping, mapping_file)
        return jsonify(result), 200
    except ValueError as e:
        logger.error(f"Link ingestion error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during link ingestion: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@main_blueprint.route('/api/v1/ingest-stream', methods=['POST'])
def ingest_stream():
    data = request.get_json()
    bucket_name = data.get('bucket_name')
    file_key = data.get('file_key')
    if not bucket_name or not file_key:
        return jsonify({"error": "Bucket name and file key are required"}), 400

    try:
        result = handle_stream_ingestion(bucket_name, file_key, client_mapping, mapping_file)
        return jsonify(result), 200
    except ValueError as e:
        logger.error(f"Stream ingestion error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Unexpected error during stream ingestion: {e}")
        return jsonify({"error": "Internal Server Error"}), 500