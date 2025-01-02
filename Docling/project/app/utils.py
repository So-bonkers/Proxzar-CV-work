# utils.py
import os
import json
import logging
from pathlib import Path
import random
from urllib.parse import urlparse

def loadConfig():
    config_file = "config.json"
    if os.path.exists(config_file):
        with open(config_file) as f:
            logging.info("Loaded configuration file.")
            return json.load(f)
    logging.warning("Configuration file not found. Using defaults.")
    return {
        "output_directory": "data/IngestedFiles",
        "temp_directory": "data/TempFiles",
        "mapping_file": "client_mapping.json",
        "supported_formats": ["pdf", "docx", "xlsx", "png", "tiff"]
    }

config = loadConfig()


def loadClientMapping():
    mapping_file = config["mapping_file"]
    if os.path.exists(mapping_file):
        with open(mapping_file) as f:
            logging.info("Loaded client mapping.")
            return json.load(f), mapping_file
    logging.warning("Client mapping file not found. Starting fresh.")
    return {}, mapping_file


def saveClientMapping(client_mapping, mapping_file):
    with open(mapping_file, "w") as f:
        json.dump(client_mapping, f)
        logging.info("Saved client mapping.")


def validateFileFormat(filename):
    _, ext = os.path.splitext(filename.lower())
    return ext[1:] in config["supported_formats"]


def generateClientID(client_mapping):
    while True:
        client_id = f"{random.randint(10000000, 99999999)}"
        if client_id not in client_mapping:
            return client_id


def getClientOutputDir(client_id):
    output_dir = Path(config["output_directory"]) / client_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
