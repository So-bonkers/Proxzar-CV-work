import json
import os
from pathlib import Path
from app.utils import loadClientMapping

def docling_to_custom_json(client_id, table_dir):
    """
    Converts a Docling JSON file to a simplified, readable JSON format with external references for images, tables, lists, and forms.

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
    input_path = Path(client_data["saved_file"]).parent / f"{client_id}-with-image-refs.json"
    output_path = f"data/convertedToJSON/{client_id}-converted.json"

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

    # Helper function to resolve references
    def resolve_ref(ref):
        ref_type, ref_index = ref.split("/")[1], int(ref.split("/")[-1])
        return docling_data.get(ref_type, [])[ref_index]

    # Process the body of the document
    for item in docling_data.get("body", {}).get("children", []):
        reference = item.get("$ref", "")

        if reference.startswith("#/texts/"):
            text_item = resolve_ref(reference)
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
            table_file = f"data/ingestedFiles/{client_id}/{client_id}-table-{index + 1}.html"

            # Add table reference
            simplified_json["content"].append({
                "header": f"Table {index + 1}",
                "subContent": [
                    {
                        "contentType": "table",
                        "source": table_file
                    }
                ]
            })

        elif reference.startswith("#/pictures/"):
            pic_item = resolve_ref(reference)
            index = int(reference.split("/")[-1])
            image_file = f"image_{index:06d}_*.png"
            image_folder = f"data/ingestedFiles/{client_id}/{client_id}-with-image-refs_artifacts"
            
            # Find the correct image file in the folder
            matching_files = [f for f in os.listdir(image_folder) if f.startswith(f"image_{index:06d}_")]
            source = os.path.join(image_folder, matching_files[0]) if matching_files else "unknown"

            # Append image content
            simplified_json["content"].append({
                "header": f"Image {index + 1}",
                "subContent": [
                    {
                        "contentType": "image",
                        "source": source
                    }
                ]
            })

        elif reference.startswith("#/groups/"):
            group = resolve_ref(reference)
            group_label = group.get("label", "group")
            group_items = []

            for child in group.get("children", []):
                child_ref = resolve_ref(child.get("$ref", ""))
                group_items.append({
                    "contentType": "text" if child_ref.get("label") != "list_item" else "list_item",
                    "source": child_ref.get("text", "")
                })

            simplified_json["content"].append({
                "header": group_label,
                "subContent": group_items
            })

    # Write the converted JSON to the output path
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_json, outfile, indent=4)

    return {"message": "Conversion successful.", "output_path": str(output_path)}

# Example of usage
client_id = "11283833"  # Replace with the actual client ID
table_dir = f"data/ingestedFiles/{client_id}"  # Directory containing the HTML table files

# Perform the conversion
result = docling_to_custom_json(client_id, table_dir)
if "error" in result:
    print(result["error"])
else:
    print(result["message"], f"File saved at: {result['output_path']}")
