import random
from pathlib import Path

def generate_client_id(client_mapping):
    """
    Generate a unique 8-digit Client ID.
    """
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            return client_id

def get_client_output_dir(output_directory, client_id):
    """
    Get the output directory for a given Client ID.
    """
    return Path(output_directory) / client_id
