# services/document_processing_service.py
import logging
from app.utils import getClientOutputDir

logger = logging.getLogger(__name__)

def handle_document_processing(client_id, client_mapping):
    if client_id not in client_mapping:
        raise ValueError(f"Client ID {client_id} not found.")

    client_data = client_mapping[client_id]
    saved_file = client_data.get("saved_file")

    if not saved_file:
        raise ValueError("Saved file not found for this client ID.")

    output_dir = getClientOutputDir(client_id)
    # Placeholder: Add actual document processing logic here
    logger.info(f"Processing document for client ID {client_id}.")
    return {"message": f"Document for client ID {client_id} processed successfully.", "output_dir": str(output_dir)}
