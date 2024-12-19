import random
import os
from pathlib import Path
from werkzeug.utils import secure_filename

def generate_client_id(client_mapping):
    """Generate a unique 8-digit Client ID.
    
    Args:
        client_mapping (dict): A dictionary mapping existing client IDs.
    
    Returns:
        str: A unique 8-digit client ID.
    """
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            return client_id

def validate_file_format(filename, supported_formats):
    """Check if the file has a supported format.
    
    Args:
        filename (str): The name of the file to check.
        supported_formats (set): A set of supported file extensions.
    
    Returns:
        bool: True if the file format is supported, False otherwise.
    """
    _, ext = os.path.splitext(filename.lower())
    return ext in supported_formats

def save_temp_file(file, temp_directory):
    """Save the uploaded file to the temporary directory.
    
    Args:
        file (FileStorage): The uploaded file.
        temp_directory (str): The path to the temporary directory.
    
    Returns:
        Path: The path to the saved temporary file.
    """
    temp_path = Path(temp_directory) / secure_filename(file.filename)
    file.save(temp_path)
    return temp_path

def move_file_to_output(temp_path, output_path):
    """Move the file from the temporary directory to the output directory.
    
    Args:
        temp_path (Path): The path to the temporary file.
        output_path (Path): The path to the output directory.
    
    Returns:
        Path: The path to the moved file in the output directory.
    """
    output_file_path = output_path / temp_path.name
    temp_path.rename(output_file_path)
    return output_file_path
