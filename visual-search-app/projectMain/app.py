import os
import shutil
from flask import Flask, request, jsonify, send_file
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
import traceback
import uuid
from PIL import Image, ImageEnhance
import numpy as np
import torch
import timm
import faiss
import pandas as pd
from torchvision import transforms
from torch.autograd import Variable
import projectMain.config as config
from projectMain.DeepImageSearch import Load_Data, Search_Setup

# Initialize Flask app and Flask-RESTful API
app = Flask(__name__)
api = Api(app)

# Define upload folder for storing uploaded images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Ensure metadata files exist for each client
METADATA_FOLDER = 'metadata-files'
os.makedirs(METADATA_FOLDER, exist_ok=True)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_transformations(image_path):
    """Applies transformations to an image: rotation, scaling, hue modification, noise addition."""
    img = Image.open(image_path).convert('RGB')
    transformations = []

    # Rotation
    for angle in [90, 180, 270]:
        transformations.append(img.rotate(angle))

    # Scaling (resize by 20%)
    w, h = img.size
    transformations.append(img.resize((int(w * 1.2), int(h * 1.2))))

    # Hue modification
    enhancer = ImageEnhance.Color(img)
    transformations.append(enhancer.enhance(1.5))

    # Adding noise
    img_np = np.array(img)
    noise = np.random.normal(0, 25, img_np.shape).astype(np.uint8)
    noisy_img = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))
    transformations.append(noisy_img)

    # Save transformed images and return their paths
    transformed_paths = []
    for idx, transformed_img in enumerate(transformations):
        transformed_path = f"{image_path.rsplit('.', 1)[0]}_trans_{idx}.jpg"
        transformed_img.save(transformed_path)
        transformed_paths.append(transformed_path)

    return transformed_paths

class RunIndex(Resource):
    """Loads the index for a specific client."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            if not client_id:
                return {"error": "Missing client_id"}, 400
            
            index_folder = os.path.join(METADATA_FOLDER, client_id)
            if not os.path.exists(index_folder):
                return {"error": f"Index not found for client {client_id}"}, 404

            return {"message": f"Index loaded for client {client_id}"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

class AddNewImage(Resource):
    """Adds a new image to an existing index after transformations."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            if 'file' not in request.files:
                return {"error": "No file uploaded"}, 400

            file = request.files['file']
            if file.filename == '':
                return {"error": "Empty filename"}, 400

            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(temp_path)

                # Apply transformations
                transformed_paths = apply_transformations(temp_path)
                transformed_paths.append(temp_path)  # Add original image

                # Add images to index
                index_folder = os.path.join(METADATA_FOLDER, client_id)
                os.makedirs(index_folder, exist_ok=True)
                search_instance = Search_Setup(image_list=[], model_name=client_id, pretrained=True)
                search_instance.add_images_to_index(transformed_paths)

                # Delete images after processing
                for img_path in transformed_paths:
                    os.remove(img_path)

                return {"message": "Images added to index"}, 200
            else:
                return {"error": "Invalid file type"}, 400
        except Exception as e:
            return {"error": str(e)}, 500

class GetSimilarImages(Resource):
    """Finds similar images for an uploaded image."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
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
    """Creates an index for a specific client."""
    def post(self):
        try:
            client_id = request.form.get('client_id')
            folder_path = request.form.get('path')

            if not os.path.exists(folder_path):
                return {"error": "Invalid folder path"}, 400

            # Load images from folder and index them
            loader = Load_Data()
            image_paths = loader.from_folder([folder_path])

            index_folder = os.path.join(METADATA_FOLDER, client_id)
            os.makedirs(index_folder, exist_ok=True)

            search_instance = Search_Setup(image_list=image_paths, model_name=client_id, pretrained=True)
            search_instance.run_index()

            return {"message": f"Index created for client {client_id}"}, 200
        except Exception as e:
            return {"error": str(e)}, 500

# Register API Endpoints
api.add_resource(RunIndex, "/runIndexWithExistingImagesOnServer")
api.add_resource(AddNewImage, "/addNewImageToIndex")
api.add_resource(GetSimilarImages, "/getSimilarImages")
api.add_resource(MakeIndex, "/makeIndex")

if __name__ == '__main__':
    app.run(debug=True)
