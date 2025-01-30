import os
import json
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from DeepImageSearch import Load_Data, Search_Setup

app = Flask(__name__)
api = Api(app)

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

class RunIndex(Resource):
    """Loads the index for a specific client."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            if not client_id:
                return {"error": "Missing client_id"}, 400
            
            if not client_exists(client_id):
                return {"error": f"Client {client_id} not found!"}, 404

            return {"message": f"Index loaded for client {client_id}"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

class AddNewImage(Resource):
    """Adds a new image to an existing index and updates image count."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            if not client_exists(client_id):
                return {"error": f"Client {client_id} does not exist!"}, 404

            if 'file' not in request.files:
                return {"error": "No file uploaded"}, 400

            file = request.files['file']
            if file.filename == '':
                return {"error": "Empty filename"}, 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(temp_path)

                # Add image to index
                search_instance = Search_Setup(image_list=[], model_name=client_id, pretrained=True)
                search_instance.add_images_to_index([temp_path])

                # Delete image after processing
                os.remove(temp_path)

                # Update client’s image count
                update_client_image_count(client_id, 1)

                return {"message": "Image added to index and count updated"}, 200
            else:
                return {"error": "Invalid file type"}, 400
        except Exception as e:
            return {"error": str(e)}, 500

class GetSimilarImages(Resource):
    """Finds similar images for an uploaded image."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            if not client_exists(client_id):
                return {"error": f"Client {client_id} does not exist!"}, 404

            if 'file' not in request.files:
                return {"error": "No file uploaded"}, 400

            file = request.files['file']
            if file.filename == '':
                return {"error": "Empty filename"}, 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(temp_path)

                # Perform image search
                search_instance = Search_Setup(image_list=[], model_name=client_id, pretrained=True)
                results = search_instance.get_similar_images(temp_path, 10)

                # Delete uploaded image after processing
                os.remove(temp_path)

                return {"similar_images": list(results.values())}, 200
            else:
                return {"error": "Invalid file type"}, 400
        except Exception as e:
            return {"error": str(e)}, 500

class MakeIndex(Resource):
    """Creates an index for a specific client and registers it in clients.json."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            folder_path = request.form.get('path')

            if not client_id or not folder_path:
                return {"error": "Client ID and folder path are required"}, 400

            if not os.path.exists(folder_path):
                return {"error": "Invalid folder path"}, 400

            if client_exists(client_id):
                return {"error": f"Client {client_id} already exists!"}, 409

            # Load images and create index
            loader = Load_Data()
            image_paths = loader.from_folder([folder_path])

            index_folder = os.path.join(METADATA_FOLDER, client_id)
            os.makedirs(index_folder, exist_ok=True)

            search_instance = Search_Setup(image_list=image_paths, model_name=client_id, pretrained=True)
            search_instance.run_index()

            # Register client
            register_client(client_id, index_folder, len(image_paths))

            return {"message": f"Index created for client {client_id}"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Register API Endpoints
api.add_resource(RunIndex, "/api/v1/runIndexWithExistingImagesOnServer")
api.add_resource(AddNewImage, "/api/v1/addNewImageToIndex")
api.add_resource(GetSimilarImages, "/api/v1/getSimilarImages")
api.add_resource(MakeIndex, "/api/v1/makeIndex")

@app.route('/')
def home():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
