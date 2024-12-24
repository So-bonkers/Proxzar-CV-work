# Document Processing Application

This project is a Flask-based web application for processing documents. It supports various functionalities such as ingesting documents via file upload, URL link, or S3 stream, extracting content from documents, and converting HTML to JSON.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Functions](#functions)
- [Configuration](#configuration)

## Features

- Ingest documents via file upload, URL link, or S3 stream.
- Extract content from ingested documents.
- Convert HTML documents to JSON format.
- Serve files for download.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/kshubhan/Proxzar-CV-work.git
    cd Proxzar-CV-work/Docling/project
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the Flask application:
    ```bash
    python run.py
    ```

2. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## API Endpoints

- `POST /api/v1/ingest`: Ingest a document via file upload.
- `POST /api/v1/ingest-link`: Ingest a document via URL link.
- `POST /api/v1/ingest-stream`: Ingest a document via S3 stream.
- `POST /api/v1/extract`: Extract content from an ingested document.
- `POST /api/v1/htmlToJson`: Convert an HTML document to JSON format.
- `GET /download/<client_id>/<filename>`: Download a processed file.

## Functions

### `app/utils.py`

- **loadConfig**: Load configuration from the config file or return default configuration.
- **loadClientMapping**: Load client mapping from the mapping file or return an empty mapping.
- **saveClientMapping**: Save client mapping to the mapping file.
- **generateClientID**: Generate a unique 8-digit Client ID.
- **validateFileFormat**: Check if the file or URL has a supported format.
- **getClientOutputDir**: Get the output directory for a given Client ID.
- **processDocument**: Process the document using Docling.
- **getS3Client**: Initialize and return an S3 client using configuration.
- **processStreamDocument**: Process a document directly from S3 stream.
- **getContentType**: Determine the content type of an HTML element and extract its data.
- **parseHTMLToJSON**: Parse an HTML file and convert its content to a JSON structure.

### `app/routes.py`

- **index**: Serve the HTML UI.
- **ingest**: Handle file ingestion.
- **ingest_link**: Handle file ingestion from a URL link.
- **ingest_stream**: Handle S3 stream ingestion.
- **extract**: Handle document extraction.
- **downloadFile**: Serve files for download.
- **htmlToJson**: Convert HTML to JSON.

### `app/__init__.py`

- **create_app**: Create and configure the Flask application.

## Configuration

The configuration file `configForApp.json` contains the following settings:

```json
{
    "output_directory": "data/IngestedFiles",
    "temp_directory": "data/TempFiles",
    "mapping_file": "client_mapping.json",
    "supported_formats": ["pdf", "docx", "xlsx", "odt", "ods", "png", "tiff", "application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/vnd.ms-excel"],
    "template_folder": "Docling\\project\\templates",
    "static_folder": "Docling\\project\\static"
}
```

Make sure to update the configuration file as needed for your environment.

## License

This project is licensed under the MIT License.