# services/ingestion_service.py
import os
from pathlib import Path
from werkzeug.utils import secure_filename
from app.utils import validateFileFormat, generateClientID, getClientOutputDir, saveClientMapping


def handle_file_ingestion(file, client_mapping, mapping_file):
    if not validateFileFormat(file.filename):
        raise ValueError(f"Unsupported file format: {file.filename}")

    client_id = generateClientID(client_mapping)
    output_path = getClientOutputDir(client_id)

    saved_file_path = output_path / secure_filename(file.filename)
    file.save(saved_file_path)

    client_mapping[client_id] = {
        "original_file": file.filename,
        "output_path": str(output_path),
        "saved_file": str(saved_file_path),
    }
    saveClientMapping(client_mapping, mapping_file)

    return {"client_id": client_id, "output_path": str(output_path)}