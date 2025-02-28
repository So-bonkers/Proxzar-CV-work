import json
import os
import re
from pathlib import Path

def docling_to_custom_json(client_id, input_dir):
    """
    Converts an HTML file to a structured JSON format including paths for extracted figures and tables.
    """
    from app.utils import loadClientMapping  

    input_path = Path(input_dir) / f"{client_id}-with-image-refs.html"
    output_path = Path("data/convertedToJSON") / f"{client_id}-converted.json"

    if not input_path.exists():
        return {"error": f"HTML file not found for client ID {client_id} at {input_path}"}

    with open(input_path, 'r', encoding='utf-8') as infile:
        content = infile.read().strip()

    if not content:
        return {"error": f"HTML file {input_path} is empty."}

    simplified_json = {
        "url": "",
        "title": input_path.name,
        "content": [],
        "figures": [],
        "tables": []
    }

    # Extract meta URL if available
    meta_url_match = re.search(r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']', content)
    simplified_json["url"] = meta_url_match.group(1) if meta_url_match else "No URL found"

    sections = []
    current_section = None
    table_count = 0
    figure_count = 0

    # Splitting content into elements
    elements = re.split(r'(<h2.*?>.*?</h2>)', content, flags=re.DOTALL)

    for element in elements:
        h2_match = re.match(r'<h2.*?>(.*?)</h2>', element, flags=re.DOTALL)
        if h2_match:
            if current_section:
                sections.append(current_section)
            current_section = {"header": h2_match.group(1).strip(), "subContent": []}
        elif current_section:
            try:
                paragraphs = re.findall(r'<p.*?>(.*?)</p>', element, flags=re.DOTALL)
                for para in paragraphs:
                    current_section["subContent"].append({
                        "contentType": "text",
                        "source": re.sub(r'<.*?>', '', para).strip()
                    })

                # Extract image references
                img_matches = re.findall(r'<img[^>]+src=["\']([^"\']+)["\']', element)
                for img_src in img_matches:
                    figure_count += 1
                    figure_path = f"data/IngestedFiles/{client_id}/{client_id}-figure-{figure_count}.png"
                    current_section["subContent"].append({
                        "contentType": "image",
                        "source": {"path": figure_path}
                    })
                    simplified_json["figures"].append(figure_path)

                # Extract tables
                table_matches = re.findall(r'<table.*?</table>', element, flags=re.DOTALL)
                for _ in table_matches:
                    table_count += 1
                    table_path = f"data/IngestedFiles/{client_id}/{client_id}-table-{table_count}.html"
                    current_section["subContent"].append({
                        "contentType": "table",
                        "source": {"path": table_path}
                    })
                    simplified_json["tables"].append(table_path)

            except Exception as e:
                continue

    if current_section:
        sections.append(current_section)

    simplified_json["content"] = sections

    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as outfile:
        json.dump(simplified_json, outfile, indent=4)

    return {"message": "Conversion successful.", "output_path": str(output_path)}
