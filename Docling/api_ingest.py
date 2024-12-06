import os
import time
import logging
import shutil
import argparse

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Directory to save ingested files
INGESTED_DOCS_DIR = r'C:\Users\DELL\Desktop\docling\ingested_docs'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'odt', 'ods', 'png', 'jpg', 'jpeg', 'tiff', 'pptx', 'html', 'asciidoc', 'md'}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_info(filepath):
    """Get file size and type."""
    try:
        file_size = os.path.getsize(filepath)
        file_extension = os.path.splitext(filepath)[1].lower()
        return file_size, file_extension
    except Exception as e:
        logging.error(f"Error retrieving file info: {filepath}, {e}")
        return None, None

def ingest_file(src_path, dest_folder):
    """Ingest a single file while preserving the file extension."""
    try:
        if not allowed_file(os.path.basename(src_path)):
            raise ValueError(f"Invalid file format: {src_path}")

        os.makedirs(dest_folder, exist_ok=True)
        file_size, file_extension = get_file_info(src_path)
        print(f"Processing file: {src_path}")
        print(f"File size: {file_size / (1024 * 1024):.2f} MB")
        print(f"File type: {file_extension}")

        timestamp = int(time.time())
        file_name, file_extension = os.path.splitext(os.path.basename(src_path))
        new_filename = f"{file_name}_{timestamp}{file_extension}"
        dest_path = os.path.join(dest_folder, new_filename)
        shutil.copy2(src_path, dest_path)
        logging.info(f"Copied {src_path} to {dest_path}")
        print(f"File successfully ingested: {dest_path}\n")
    except Exception as e:
        logging.error(f"Error ingesting file: {src_path}, {e}")
        print(f"Error ingesting file: {e}")

def ingest_folder(src_folder, dest_folder):
    """Recursively ingest files from a folder."""
    try:
        for item in os.listdir(src_folder):
            src_path = os.path.join(src_folder, item)
            if os.path.isdir(src_path):
                ingest_folder(src_path, dest_folder)
            elif allowed_file(item):
                ingest_file(src_path, dest_folder)
            else:
                logging.warning(f"Invalid file format: {src_path}")
                print(f"Skipping invalid file format: {src_path}")
    except Exception as e:
        logging.error(f"Error ingesting folder: {src_folder}, {e}")
        print(f"Error ingesting folder: {e}")

def main():
    """Main function to handle command-line input."""
    parser = argparse.ArgumentParser(description="Ingest files or folders.")
    parser.add_argument("path", help="Path to the file or folder to ingest")
    args = parser.parse_args()

    src_path = args.path

    if os.path.isfile(src_path):
        ingest_file(src_path, INGESTED_DOCS_DIR)
    elif os.path.isdir(src_path):
        ingest_folder(src_path, INGESTED_DOCS_DIR)
    else:
        print(f"Invalid path: {src_path}")

if __name__ == "__main__":
    main()
