import os
from pathlib import Path
from werkzeug.utils import secure_filename

def validate_file_format(filename, supported_formats):
    """
    Check if the file has a supported format.
    """
    _, ext = os.path.splitext(filename.lower())
    return ext in supported_formats

def save_temp_file(file, temp_directory):
    """
    Save the uploaded file to the temporary directory.
    """
    temp_path = Path(temp_directory) / secure_filename(file.filename)
    file.save(temp_path)
    return temp_path
