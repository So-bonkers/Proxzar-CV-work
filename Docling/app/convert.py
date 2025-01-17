import json
import os
from pathlib import Path
from app.utils import loadClientMapping

global_client_id = ""

def docling_to_custom_json(client_id, table_dir):
    """
    Converts a Docling JSON file to a simplified, readable JSON format with external table references.

    Args:
        client_id (str): The client ID to locate the Docling JSON file.
        table_dir (str): Directory where external table HTML files are located.

    Returns:
        dict: A dictionary containing the status and the output path.
    """
    # Load the client mapping
    client_mapping, _ = loadClientMapping()

    # Retrieve the Docling JSON file path based on client ID
    if client_id not in client_mapping:
        return {"error": f"Client ID {client_id} not found in mapping."}

    client_data = client_mapping[client_id]
    global_client_id = client_mapping[client_id]
    input_path = Path(client_data["saved_file"]).parent / f"{client_id}-with-image-refs.json"
    output_path = Path(client_data["output_path"]) / f"{client_id}-converted.json"

    if not input_path.exists():
        return {"error": f"Docling JSON file not found for client ID {client_id}"}

    with open(input_path, 'r', encoding='utf-8') as infile:
        docling_data = json.load(infile)

    # Initialize the output JSON structure
    simplified_json = {
        "url": docling_data.get("origin", {}).get("filename", ""),
        "title": docling_data.get("name", ""),
        "content": []
    }

    # Process the body of the document
    for item in docling_data.get("body", {}).get("children", []):
        reference = item.get("$ref", "")
        
        if reference.startswith("#/texts/"):
            text_data = docling_data.get("texts", [])
            index = int(reference.split("/")[-1])
            text_item = text_data[index]
            section_header = text_item.get("label", "unknown")

            # Append the text content
            simplified_json["content"].append({
                "header": section_header,
                "subContent": [
                    {
                        "contentType": "text",
                        "source": text_item.get("text", "")
                    }
                ]
            })

        elif reference.startswith("#/tables/"):
            index = int(reference.split("/")[-1])
            table_file = f"{client_id}-table-{index + 1}.html"
            table_path = Path(table_dir) / table_file

            # Add table reference if the file exists
            if table_path.exists():
                simplified_json["content"].append({
                    "header": f"Table {index + 1}",
                    "subContent": [
                        {
                            "contentType": "table",
                            "source": str(table_file)
                        }
                    ]
                })

        elif reference.startswith("#/pictures/"):
            pic_data = docling_data.get("pictures", [])
            index = int(reference.split("/")[-1])
            pic_item = pic_data[index]

            # Append image content
            simplified_json["content"].append({
                "header": f"Image {index + 1}",
                "subContent": [
                    {
                        "contentType": "image",
                        "source": pic_item.get("uri", "unknown")
                    }
                ]
            })

    # Write the converted JSON to the output path
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_json, outfile, indent=4)

    return {"message": "Conversion successful.", "output_path": str(output_path)}

# Example of usage
client_id = global_client_id
table_dir = f"data/ingestedFiles/{client_id}"  # Directory containing the HTML table files

# Perform the conversion
result = docling_to_custom_json(client_id, table_dir)
if "error" in result:
    print(result["error"])
else:
    print(result["message"], f"File saved at: {result['output_path']}")
