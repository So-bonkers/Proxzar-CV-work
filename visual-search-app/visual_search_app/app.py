################################################################################################################
# *Purpose: - The code in this file is used to create a Flask application that allows users to upload an image
#             and retrieve similar images from a dataset.
#           - The code uses the DeepImageSearch module to index and search for similar images. The underlying  
#             model that DeepImageSearch uses is VGG19. It generates a feature vector for each image in the 
#             dataset and uses these feature vectors to find similar images.
# It is a flask app that runs a server and allows users to upload an image and retrieve similar images from a 
# dataset.
# The flask app is deployed as a web application that allows users to upload an image and retrieve similar
# images' indices from the pickle file which is not accessible to the user.
# *The endpoints of the Flask application are as follows:
#   - /api/post: Handles the POST request for the /api/post endpoint. Acts as a test endpoint to check if the 
#                POST method is working.
#   - /query-image: Renders the upload.html template for querying an image.
#   - /inferred-indices: Handles the POST request for inferring similar image indices. It retrieves similar 
#                        images based on the uploaded file and renders the results.html template with the 
#                        inferred indices.
#
# Created By - Shubhankar Kamthankar
## Date - 22nd March 2024
## Current Version on LIVE - 1.0
#
# History of Changes:
# Date  Changed By  Version  Change Description
# 22nd March 2024  Shubhankar Kamthankar  1.0  Initial Version
# Future Changes go here
#
#
# ---------
################################################################################################################

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Import necessary libraries
import numpy as np
from flask import Flask, request, render_template, jsonify
from DeepImageSearch import Load_Data, Search_Setup
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the indexed data (assuming you have already indexed your images)
image_list = Load_Data().from_folder([r'C:\Users\kshubhan\Downloads\Proxzar-CV-work-updated\Proxzar-CV-work-main\Proxzar\data'])

# Set up the search engine
search_engine = Search_Setup(image_list=image_list, model_name='vgg19', pretrained=True, image_count=34000)

# Get the metadata of the indexed images
metadata = search_engine.get_image_metadata_file()

@app.route('/api/post', methods=['POST'])
def api_post():
    """
    Handles the POST request for the /api/post endpoint.

    Returns:
        str: A response indicating that the POST method is working.
    """
    return 'POST method is working'

@app.route('/query-image', methods=['GET'])
def index():
    """
    Renders the upload.html template for querying an image.
    """
    return render_template('upload.html')

@app.route('/inferred-indices', methods=['POST'])
def inferred_indices():
    """
    Handles the POST request for inferring similar image indices.

    Returns:
        render_template: Renders the results.html template with the inferred indices.
    """
    # Check if the file is present in the request
    if 'image' not in request.files:
        return 'No file uploaded', 400

    file = request.files['image']

    # Check if the file has a valid filename
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file temporarily
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Get similar images
    """
    This line calls the `get_similar_images` function from the `search_engine` module to retrieve similar images based on the uploaded file.
    """
    similar_images = search_engine.get_similar_images(image_path=file_path, number_of_images=10)

    # Remove the temporary file
    os.remove(file_path)

    # Extract the indices from similar_images
    indices = list(similar_images.keys())

    return render_template('results.html', indices=indices)

if __name__ == '__main__':
    app.run(debug=True)