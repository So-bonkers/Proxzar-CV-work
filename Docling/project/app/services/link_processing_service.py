# services/link_ingestion_service.py
import logging
from pathlib import Path
from app.utils import generateClientID, getClientOutputDir, saveClientMapping
import requests
from werkzeug.utils import secure_filename

logger = logging.getLogger(__name__)

def handle_link_ingestion(file_link, client_mapping, mapping_file):
    try:
        response = requests.get(file_link, stream=True)
        response.raise_for_status()

        filename = secure_filename(Path(file_link).name)
        client_id = generateClientID(client_mapping)
        output_dir = getClientOutputDir(client_id)
        saved_file_path = output_dir / filename

        with open(saved_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        client_mapping[client_id] = {
            "original_file": filename,
            "output_path": str(output_dir),
            "saved_file": str(saved_file_path),
        }
        saveClientMapping(client_mapping, mapping_file)

        return {"client_id": client_id, "output_path": str(output_dir)}
    except Exception as e:
        logger.error(f"Error during link ingestion: {e}")
        raise ValueError("Failed to ingest file from link.")