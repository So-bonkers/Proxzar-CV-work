import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import io
import matplotlib.pyplot as plt
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from DeepImageSearch import Load_Data, Search_Setup

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

METADATA_FOLDER = 'metadata-files'
os.makedirs(METADATA_FOLDER, exist_ok=True)

CLIENTS_FILE = "clients.json"

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------- CLIENT MANAGEMENT UTILITIES ----------------

def load_clients():
    """Loads the clients.json file."""
    if not os.path.exists(CLIENTS_FILE):
        with open(CLIENTS_FILE, "w") as f:
            json.dump({"clients": {}}, f)
    with open(CLIENTS_FILE, "r") as f:
        return json.load(f)

def save_clients(data):
    """Saves the updated client data to clients.json."""
    with open(CLIENTS_FILE, "w") as f:
        json.dump(data, f, indent=4)

def register_client(client_id, index_path, image_count):
    """Registers a new client in clients.json."""
    data = load_clients()
    if client_id in data["clients"]:
        return False  # Client already exists
    data["clients"][client_id] = {
        "index_path": index_path,
        "image_count": image_count,
        "created_at": datetime.now().isoformat()
    }
    save_clients(data)
    return True

def update_client_image_count(client_id, new_images_count):
    """Updates the image count when new images are added."""
    data = load_clients()
    if client_id in data["clients"]:
        data["clients"][client_id]["image_count"] += new_images_count
        save_clients(data)

def client_exists(client_id):
    """Checks if a client is registered."""
    data = load_clients()
    return client_id in data["clients"]

# --------------- API ENDPOINTS ----------------

@app.route('/api/v1/makeIndex', methods=['POST'])
def make_index():
    """Creates an index for a specific client but keeps the model as vgg19 while saving files prefixed with client ID."""
    try:
        client_id = request.form.get('client_id')
        folder_path = request.form.get('path')

        if not client_id or not folder_path:
            return jsonify({"error": "Client ID and folder path are required"}), 400

        if not os.path.exists(folder_path):
            return jsonify({"error": "Invalid folder path"}), 400

        if client_exists(client_id):
            return jsonify({"error": f"Client {client_id} already exists!"}), 409

        # Load images from folder
        loader = Load_Data()
        image_paths = loader.from_folder([folder_path])

        #   Ensure correct folder naming
        index_folder = os.path.join(METADATA_FOLDER, f"{client_id}_vgg19")
        os.makedirs(index_folder, exist_ok=True)

        # Use "vgg19" for the model but save with client ID
        search_instance = Search_Setup(image_list=image_paths, client_id=client_id, model_name="vgg19", pretrained=True)
        search_instance.run_index()

        # Register client
        register_client(client_id, index_folder, len(image_paths))

        return jsonify({"message": f"Index created for client {client_id}, model used: vgg19, saved as {client_id}_vgg19"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/v1/runIndexWithExistingImagesOnServer', methods=['POST'])
def run_index():
    """Loads the index for a specific client."""
    try:
        client_id = request.form.get('client_id')

        if not client_id:
            return jsonify({"error": "Missing client_id"}), 400

        if not client_exists(client_id):
            return jsonify({"error": f"Client {client_id} not found!"}), 404

        return jsonify({"message": f"Index loaded for client {client_id}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/addNewImageToIndex', methods=['POST'])
def add_new_image():
    """Adds a new image to an existing index and updates image count."""
    try:
        client_id = request.form.get('client_id')
        if not client_exists(client_id):
            return jsonify({"error": f"Client {client_id} does not exist!"}), 404

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(temp_path)

            # Add image to index
            search_instance = Search_Setup(image_list=[], client_id=client_id ,model_name="vgg19", pretrained=True)
            search_instance.add_images_to_index([temp_path])

            # Delete image after processing
            os.remove(temp_path)

            # Update client’s image count
            update_client_image_count(client_id, 1)

            return jsonify({"message": "Image added to index and count updated"}), 200
        else:
            return jsonify({"error": "Invalid file type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/getSimilarImages', methods=['POST'])
def get_similar_images():
    """Finds similar images and returns the plot as an image response."""
    try:
        client_id = request.form.get('client_id')
        if not client_id:
            return jsonify({"error": "Missing client_id"}), 400

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        # Use Search_Setup to find and plot similar images
        search_instance = Search_Setup(image_list=[], client_id=client_id, model_name="vgg19", pretrained=True)
        plt.figure()  # Create a new figure
        search_instance.plot_similar_images(temp_path, 10)

        # Save the figure to a temporary file
        plot_path = os.path.join(UPLOAD_FOLDER, f"similar_images_{client_id}.png")
        plt.savefig(plot_path)
        plt.close()  # Close the figure to free memory

        # Delete uploaded image after processing
        os.remove(temp_path)

        # Return the plot as an image response
        return send_file(plot_path, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
