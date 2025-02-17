import json
import os
import re
from pathlib import Path
from app.utils import loadClientMapping

def docling_to_custom_json(client_id, input_dir):
    """
    Converts an HTML file to a structured JSON format with external references for images, tables, lists, and headers.

    Args:
        client_id (str): The client ID to locate the HTML file.
        input_dir (Path): The directory containing the input HTML file.

    Returns:
        dict: A dictionary containing the status and the output path.
    """
    input_path = Path(input_dir) / str(client_id) / f"{client_id}-with-image-refs.html"
    output_path = f"data/convertedToJSON/{client_id}-converted.json"

    if not input_path.exists():
        print(f"Error: HTML file not found at {input_path}")
        return {"error": f"HTML file not found for client ID {client_id} at {input_path}"}

    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.read().strip()

    if not content:
        print(f"Error: HTML file {input_path} is empty.")
        return {"error": f"HTML file {input_path} is empty."}

    # Initialize the output JSON structure
    simplified_json = {
        "url": "",
        "title": input_path.name,
        "content": []
    }

    # Extract URL if available in meta tags
    meta_url_match = re.search(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']', content)
    if meta_url_match:
        simplified_json["url"] = meta_url_match.group(1)

    # Process the document elements
    sections = []
    current_section = None
    table_count = 0  # Tracks actual table count
    h2_count = 0  # Tracks consecutive H2 tags

    # Splitting the HTML content into elements while preserving structure
    elements = re.split(r'(<h2.*?>.*?</h2>)', content, flags=re.DOTALL)

    for element in elements:
        h2_match = re.match(r'<h2.*?>(.*?)</h2>', element, flags=re.DOTALL)
        if h2_match:
            # If a new H2 tag appears and the previous section exists, save it
            if current_section:
                sections.append(current_section)
            
            # Track consecutive H2 tags
            if current_section and not current_section["subContent"]:
                current_section["header"] += " " + h2_match.group(1).strip()
            else:
                current_section = {"header": h2_match.group(1).strip(), "subContent": []}
                h2_count += 1
        elif current_section:
            try:
                paragraphs = re.findall(r'<p.*?>(.*?)</p>', element, flags=re.DOTALL)
                for para in paragraphs:
                    current_section["subContent"].append({
                        "contentType": "text",
                        "source": re.sub(r'<.*?>', '', para).strip()
                    })
                
                img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', element)
                for img_src in img_matches:
                    caption_match = re.search(r'<figcaption.*?>(.*?)</figcaption>', content, flags=re.DOTALL)
                    caption = caption_match.group(1).strip() if caption_match else ""
                    current_section["subContent"].append({
                        "contentType": "image",
                        "source": {"caption": caption, "path": img_src.replace("\\", "/")}
                    })
                
                table_matches = re.findall(r'<table.*?</table>', element, flags=re.DOTALL)
                for _ in table_matches:
                    caption_match = re.search(r'<caption.*?>(.*?)</caption>', element, flags=re.DOTALL)
                    table_count += 1  # Increment table count for unique table references
                    table_path = f"data/ingestedFiles/{client_id}/{client_id}-table-{table_count}.html"
                    caption = caption_match.group(1).strip() if caption_match else ""
                    current_section["subContent"].append({
                        "contentType": "table",
                        "source": {"caption": caption, "path": table_path}
                    })
                
                list_matches = re.findall(r'<ul.*?</ul>|<ol.*?</ol>', element, flags=re.DOTALL)
                for list_element in list_matches:
                    list_items = re.findall(r'<li.*?>(.*?)</li>', list_element, flags=re.DOTALL)
                    current_section["subContent"].append({
                        "contentType": "list",
                        "items": [item.strip() for item in list_items]
                    })
            except Exception as e:
                print(f"Error processing element: {e}")
                continue
    
    if current_section:
        sections.append(current_section)

    simplified_json["content"] = sections

    # Write the converted JSON to the output path
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_json, outfile, indent=4)

    return {"message": "Conversion successful.", "output_path": str(output_path)}

# Example of usage
client_id = "11283833"  # Replace with the actual client ID
input_dir = "data/ingestedFiles"  # Directory containing the HTML file

# Perform the conversion
result = docling_to_custom_json(client_id, input_dir)
if "error" in result:
    print(result["error"])
else:
    print(result["message"], f"File saved at: {result['output_path']}")
