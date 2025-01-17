import logging
import mimetypes
import requests
from flask import Blueprint, render_template, request, jsonify, send_from_directory
from pathlib import Path
from werkzeug.utils import secure_filename
from app.convert import docling_to_custom_json
from app.utils import (
    generateClientID,
    validateFileFormat,
    getClientOutputDir,
    processDocument,
    loadClientMapping,
    saveClientMapping,
    parseHTMLToJSON,
    loadConfig,
    processStreamDocument
)
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config.json"
config = loadConfig()

# Create a Blueprint for the main routes
main_blueprint = Blueprint("main", __name__)

# Load client mapping from file
client_mapping, mapping_file = loadClientMapping()

@main_blueprint.route('/')
def index():
    """
    Serve the HTML UI.
    
    Returns:
        str: Rendered HTML template for the index page.
    """
    logger.info("Rendering index page.")
    return render_template('index.html')

@main_blueprint.route('/api/v1/ingest', methods=['POST'])
def ingest():
    """
    Handle file ingestion.
    
    Returns:
        Response: JSON response with client ID and output path.
    """
    # Get the file from the request
    file = request.files.get('file')
    if not file:
        logger.error("No file provided for ingestion.")
        return jsonify({"error": "No file provided"}), 400

    # Validate the file format
    if not validateFileFormat(file.filename):
        logger.error(f"Unsupported file format: {file.filename}")
        return jsonify({"error": f"Unsupported file format: {file.filename}"}), 400

    # Generate a unique client ID
    client_id = generateClientID(client_mapping)
    # Get the output directory for the client
    output_path = getClientOutputDir(client_id)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the file to the output directory
    saved_file_path = output_path / secure_filename(file.filename)
    file.save(saved_file_path)

    # Update the client mapping with the new file information
    client_mapping[client_id] = {
        "original_file": file.filename,
        "output_path": str(output_path),
        "saved_file": str(saved_file_path),
    }
    saveClientMapping(client_mapping, mapping_file)

    logger.info(f"File {file.filename} ingested successfully with Client ID {client_id}.")
    return jsonify({"client_id": client_id, "output_path": str(output_path)})

@main_blueprint.route('/api/v1/ingest-link', methods=['POST'])
def ingest_link():
    """
    Handle file ingestion from a URL link.
    
    Returns:
        Response: JSON response with client ID and output path.
    """
    data = request.get_json()
    file_link = data.get('file_link', '').strip()

    if not file_link:
        logger.error("No file link provided.")
        return jsonify({"error": "No file link provided"}), 400

    logger.info(f"Received file link: {file_link}")

    temp_directory = Path(config["temp_directory"])
    temp_directory.mkdir(parents=True, exist_ok=True)

    try:
        response = requests.get(file_link, stream=True, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Get filename from Content-Disposition header or URL
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"')
        else:
            filename = os.path.basename(file_link)
            if not filename:
                filename = "downloaded_file" # Default name if no filename can be extracted

        filename = secure_filename(filename) # Sanitize the filename
        temp_file_path = temp_directory / filename

        with open(temp_file_path, 'wb') as temp_file:
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)

        # Validate file format based on downloaded file
        mime_type, _ = mimetypes.guess_type(temp_file_path)

        if mime_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "image/png", "image/tiff", "image/jpeg"]: #added jpeg
            logger.error(f"Unsupported file format: {mime_type}")
            temp_file_path.unlink()
            return jsonify({"error": f"Unsupported file format: {mime_type}"}), 400
        
        # ... (rest of the file processing - client ID generation, moving file, etc.)
        client_id = generateClientID(client_mapping)
        output_path = getClientOutputDir(client_id)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_file_path = output_path / temp_file_path.name
        temp_file_path.rename(saved_file_path)

        client_mapping[client_id] = {
                "original_file": filename,
                "output_path": str(output_path),
                "saved_file": str(saved_file_path),
            }
        saveClientMapping(client_mapping, mapping_file)

        logger.info(f"File successfully ingested for Client ID: {client_id}")
        return jsonify({"client_id": client_id, "output_path": str(output_path)})

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching URL: {e}")
        return jsonify({"error": f"Error fetching URL: {e}"}), 500
    except Exception as e:
        logger.error(f"Error processing file link: {e}")
        return jsonify({"error": f"Error processing file link: {str(e)}"}), 500
    
@main_blueprint.route('/api/v1/ingest-stream', methods=['POST'])
def ingest_stream():
    """
    Handle S3 stream ingestion.
    
    Returns:
        Response: JSON response with client ID and output path.
    """
    data = request.get_json()
    bucket_name = data.get('bucket_name')
    file_key = data.get('file_key')
    
    if not bucket_name or not file_key:
        logger.error("Bucket name or file key missing from stream ingestion request.")
        return jsonify({"error": "Both bucket_name and file_key are required"}), 400
    
    try:
        # Generate a unique client ID
        client_id = generateClientID(client_mapping)
        # Get the output directory for the client
        output_path = getClientOutputDir(client_id)
        output_path.mkdir(parents=True, exist_ok=True)

        # Process the stream
        result = processStreamDocument(bucket_name, file_key, output_path, client_id)
        if "error" in result:
            return jsonify({"error": result["error"]}), 500

        # Update the client mapping with the stream information
        client_mapping[client_id] = {
            "original_file": file_key,
            "output_path": str(output_path),
            "bucket_name": bucket_name,
            "is_stream": True
        }
        saveClientMapping(client_mapping, mapping_file)

        logger.info(f"Stream from bucket {bucket_name}, file {file_key} ingested successfully with Client ID {client_id}.")
        return jsonify({
            "client_id": client_id, 
            "output_path": str(output_path),
            "message": "Stream document processed successfully"
        })

    except Exception as e:
        logger.error(f"Error processing stream document: {e}")
        return jsonify({"error": str(e)}), 500
    
@main_blueprint.route('/api/v1/extract', methods=['POST'])
def extract():
    """
    Handle document extraction.
    
    Returns:
        Response: JSON response with extraction result.
    """
    # Get the client ID from the request
    client_id = request.form.get('client_id')
    if not client_id:
        logger.error("Client ID is missing from extraction request.")
        return jsonify({"error": "Client ID is required"}), 400

    # Check if the client ID exists in the mapping
    if client_id not in client_mapping:
        logger.error(f"Client ID {client_id} not found in mapping.")
        return jsonify({"error": f"Client ID {client_id} not found"}), 404

    client_data = client_mapping[client_id]
    saved_file_path = Path(client_data["saved_file"])
    output_path = Path(client_data["output_path"])

    # Check if the saved file exists
    if not saved_file_path.exists():
        logger.error(f"Original file not found: {saved_file_path}")
        return jsonify({"error": f"Original file not found: {saved_file_path}"}), 404

    # Process the document
    result = processDocument(saved_file_path, output_path, global_client_id=client_id)
    if "error" in result:
        logger.error(f"Error during extraction: {result['error']}")
        return jsonify({"error": result["error"]}), 500

    logger.info(f"Document extracted successfully for Client ID {client_id}.")
    return jsonify({
        "message": "Document extracted successfully",
        "output_directory": result["output_dir"]
    })

@main_blueprint.route('/download/<client_id>/<filename>')
def downloadFile(client_id, filename):
    """
    Serve files for download.
    
    Args:
        client_id (str): The client ID.
        filename (str): The name of the file to download.
    
    Returns:
        Response: The file to be downloaded.
    """
    logger.info(f"Serving download request for Client ID {client_id}, file {filename}.")
    # Get the output directory for the client
    output_dir = getClientOutputDir(client_id)
    return send_from_directory(output_dir, filename)

@main_blueprint.route('/api/v1/htmlToJson', methods=['POST'])
def htmlToJson():
    """
    Convert HTML to JSON.
    
    Returns:
        Response: JSON response with conversion result.
    """
    try:
        # Get the client ID from the request
        client_id = request.json.get('client_id')
        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        # Define the file paths
        html_file_path = os.path.join('data', 'IngestedFiles', client_id, f'{client_id}-with-image-refs.html')
        json_file_path = os.path.join('data', 'convertedToJSON', f'{client_id}.json')

        # Check if the HTML file exists
        if not os.path.exists(html_file_path):
            return jsonify({"error": f"HTML file for client {client_id} not found"}), 404
        
        # Convert HTML to JSON
        parseHTMLToJSON(html_file_path, json_file_path)

        return jsonify({"message": f"HTML file converted to JSON for client {client_id}", "json_file": json_file_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@main_blueprint.route('/api/v1/json-conversion', methods=['POST'])
def json_conversion():
    """
    Convert a Docling JSON file to a simplified JSON format.

    Returns:
        Response: JSON response with the status and output file path.
    """
    try:
        # Get the client ID from the request
        client_id = request.json.get('client_id')
        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        # Directory for HTML tables
        table_dir = config.get("output_directory", "data/IngestedFiles")

        # Perform the conversion
        result = docling_to_custom_json(client_id, table_dir)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404

        return jsonify({"message": result["message"], "output_path": result["output_path"]}), 200

    except Exception as e:
        logger.error(f"Error in JSON conversion: {e}")
        return jsonify({"error": str(e)}), 500


@main_blueprint.route('/api/v1/json-conversion', methods=['POST'])
def json_conversion():
    """
    Convert a Docling JSON file to a simplified JSON format.

    Returns:
        Response: JSON response with the status and output file path.
    """
    try:
        # Get the client ID from the request
        client_id = request.json.get('client_id')
        if not client_id:
            return jsonify({"error": "Client ID is required"}), 400

        # Directory for HTML tables
        table_dir = config.get("output_directory", "data/IngestedFiles")

        # Perform the conversion
        result = docling_to_custom_json(client_id, table_dir)

        if "error" in result:
            return jsonify({"error": result["error"]}), 404

        return jsonify({"message": result["message"], "output_path": result["output_path"]}), 200

    except Exception as e:
        logger.error(f"Error in JSON conversion: {e}")
        return jsonify({"error": str(e)}), 500
