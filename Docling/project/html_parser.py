from bs4 import BeautifulSoup
import json

def parse_html_to_json(html_file, output_json):
    """
    Parse an HTML file and convert its content to a JSON structure.
    
    Args:
        html_file (str): The path to the input HTML file.
        output_json (str): The path to the output JSON file.
    """
    with open(html_file, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')
    
    json_data = []
    title = soup.title.string if soup.title else ""
    url = soup.find('link', rel="canonical")['href'] if soup.find('link', rel="canonical") else ""
    
    # Main JSON object
    document_data = {
        "url": url,
        "title": title,
        "content": []
    }
    
    # Process headers and their content
    headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    for header in headers:
        header_data = {
            "header": header.text.strip(),
            "subContent": []
        }
        
        # Extract sibling content until the next header
        for sibling in header.find_next_siblings():
            if sibling.name and sibling.name.startswith('h'):
                break
            content_type = get_content_type(sibling)
            header_data["subContent"].append(content_type)
        
        document_data["content"].append(header_data)
    
    json_data.append(document_data)
    
    # Save to JSON file
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(json_data, json_file, indent=4)

def get_content_type(element):
    """
    Determine the content type of an HTML element and extract its data.
    
    Args:
        element (Tag): A BeautifulSoup Tag object representing an HTML element.
    
    Returns:
        dict: A dictionary containing the content kind and its source data.
    """
    if element.name == 'p':
        return {"contentType": "paragraph", "source": element.text.strip()}
    elif element.name in ['ul', 'ol']:
        list_items = [li.text.strip() for li in element.find_all('li')]
        return {"contentType": "list", "source": list_items}
    elif element.name == 'table':
        rows = []
        for tr in element.find_all('tr'):
            row = []
            for cell in tr.find_all(['th', 'td']):
                row.append(cell.text.strip())
            rows.append(row)
        return {"contentType": "table", "source": rows}
    elif element.name == 'img':
        return {"contentType": "image", "source": element['src']}
    return {}

# Usage example
parse_html_to_json(r'C:\Users\kshubhan\Documents\GitHub\Proxzar-CV-work\Docling\appTesting\data\ingestedFiles\34531891\2412.13195v1-with-image-refs.html', 'output.json')
