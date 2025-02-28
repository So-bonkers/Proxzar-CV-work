from flask import Blueprint, request, jsonify, render_template, session
import logging
import requests
import os
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
from app.utils import generateClientID, getClientOutputDir, saveClientMapping, processDocument, loadClientMapping, loadConfig

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
    """Render the homepage."""
    return render_template('index.html')

# Ensure JWT is available before request
@main_blueprint.before_request
def ensure_jwt_token():
    if 'jwt_token' not in session:
        session['jwt_token'] = create_access_token(identity="user")

@main_blueprint.route('/api/v1/authenticate', methods=['POST'])
def authenticate():
    """Generate and return a JWT token."""
    token = create_access_token(identity="user")
    session['jwt_token'] = token  # ✅ Store token in session
    logger.info(f"Generated JWT token: {token}")
    return jsonify(access_token=token), 200

@main_blueprint.route('/api/v1/process-file', methods=['POST'])
@jwt_required()
def process_file():
    """Ingest, extract, and convert file content in a single request."""
    user = get_jwt_identity()
    logger.info(f"Authenticated user: {user}")

    data = request.get_json()
    file_link = data.get("file_link")

    if not file_link:
        return jsonify({"error": "File link is required"}), 400

    # Generate client ID
    client_id = generateClientID(client_mapping)
    logger.info(f"Generated Client ID: {client_id}")
    
    # Save client ID mapping
    client_mapping[client_id] = file_link
    saveClientMapping(client_mapping, mapping_file)

    # Ensure output directory exists
    output_dir = getClientOutputDir(client_id)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download the file
    logger.info(f"Downloading file from {file_link}")
    try:
        response = requests.get(file_link)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        return jsonify({"error": f"Failed to download file: {e}"}), 500

    # Step 2: Save file
    filename = os.path.join(output_dir, os.path.basename(file_link))
    with open(filename, 'wb') as file:
        file.write(response.content)
    logger.info(f"File saved at {filename}")

    # Step 3: Process Document
    process_result = processDocument(filename, output_dir, client_id)
    if "error" in process_result:
        logger.error(f"Processing failed for Client ID {client_id}: {process_result['error']}")
        return jsonify({"error": process_result["error"]}), 500

    logger.info(f"Processing successful for Client ID: {client_id}")

    return jsonify({
        "message": "Full process completed successfully",
        "client_id": client_id,
        "output_json": process_result["output_json"]
    })