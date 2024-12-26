from flask import Flask, request, jsonify, render_template
import os
import json
import random
from DeepImageSearch.DeepImageSearch import Load_Data, Search_Setup
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploaded_images')
METADATA_FOLDER = os.path.join(os.getcwd(), 'metadata-files')
CLIENTS_JSON = os.path.join(os.getcwd(), 'clients.json')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(METADATA_FOLDER, exist_ok=True)

# Initialize clients.json if not present
if not os.path.exists(CLIENTS_JSON):
    with open(CLIENTS_JSON, 'w') as f:
        json.dump({}, f)

# Flask configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['METADATA_FOLDER'] = METADATA_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/registerClient', methods=['POST'])
def register_client():
    try:
        data = request.json
        client_name = data.get('client_name')
        image_path = data.get('image_path')

        if not client_name or not image_path:
            return jsonify({"error": "Missing client_name or image_path."}), 400

        with open(CLIENTS_JSON, 'r') as f:
            clients = json.load(f)

        # Check if the client already exists
        for client_id, info in clients.items():
            if info['client_name'] == client_name and info['image_path'] == image_path:
                return jsonify({"client_id": client_id}), 200

        # Generate a new 5-digit client ID
        client_id = str(random.randint(10000, 99999))
        while client_id in clients:
            client_id = str(random.randint(10000, 99999))

        # Add new client
        clients[client_id] = {
            "client_name": client_name,
            "image_path": image_path
        }

        with open(CLIENTS_JSON, 'w') as f:
            json.dump(clients, f)

        return jsonify({"client_id": client_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/runIndexWithExistingImagesOnServer', methods=['POST'])
def run_index():
    try:
        data = request.json
        client_id = data.get('client_id')
        folder = data.get('folder')

        if not client_id or not folder:
            return jsonify({"error": "Missing client_id or folder."}), 400

        # Use raw string and normalize path
        folder = os.path.normpath(folder)
        print(f"Checking folder path: {folder}")
        
        if not os.path.exists(folder):
            return jsonify({"error": f"Image folder does not exist at: {folder}"}), 404

        images = Load_Data().from_folder([folder])
        print(f"Found {len(images)} images in folder")
        
        search = Search_Setup(image_list=images, model_name=client_id)
        search.run_index()

        return jsonify({"message": f"Indexing completed for client_id: {client_id}. Processed {len(images)} images"}), 200
    except Exception as e:
        print(f"Error details: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/addNewImageToIndex', methods=['POST'])
def add_image():
    try:
        client_id = request.form.get('client_id')
        image_file = request.files.get('image')

        if not client_id or not image_file:
            return jsonify({"error": "Missing client_id or image."}), 400

        client_folder = os.path.join(app.config['UPLOAD_FOLDER'], client_id)
        os.makedirs(client_folder, exist_ok=True)

        image_path = os.path.join(client_folder, secure_filename(image_file.filename))
        image_file.save(image_path)

        search = Search_Setup(image_list=[], model_name=client_id)
        search.add_images_to_index([image_path])

        return jsonify({"message": "Image added to index for client_id: {}.".format(client_id)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getSimilarImages', methods=['POST'])
def get_similar_images():
    try:
        client_id = request.form.get('client_id')
        image_file = request.files.get('image')
        num_images = int(request.form.get('num_images', 5))

        if not client_id or not image_file:
            return jsonify({"error": "Missing client_id or image."}), 400

        query_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        image_file.save(query_path)

        search = Search_Setup(image_list=[], model_name=client_id)
        similar_images = search.get_similar_images(query_path, number_of_images=num_images)

        return jsonify({"similar_images": similar_images}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
