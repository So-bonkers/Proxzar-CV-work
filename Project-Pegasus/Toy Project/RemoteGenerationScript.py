import json
import ollama
import re
from tqdm import tqdm

# Load input JSON file
input_file = "QG_custom_range.json_1.json"
output_file = "QG_summarized_range_1.json"

# Function to generate summary using DeepSeek-R1:1.5B
def generate_summary(context):
    prompt = f"Summarize the following text under 50 words:\n\n{context}"
    response = ollama.chat(model="deepseek-r1", messages=[{"role": "user", "content": prompt}])
    if "message" in response:
        summary = response["message"]["content"]
        # Remove content within <think>...</think> tags
        summary_cleaned = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        return summary_cleaned
    return "Summary not available"

# Load input JSON
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Ensure we don't overwrite existing content; open file in append mode
try:
    with open(output_file, "r", encoding="utf-8") as f:
        summarized_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    summarized_data = {"results": []}

# Get existing indices to avoid duplicates
existing_indices = {entry["index"] for entry in summarized_data["results"]}

# Open file in append mode for progressive writing
with open(output_file, "a", encoding="utf-8") as f:
    if not summarized_data["results"]:
        f.write("{\n  \"results\": [\n")  # Start JSON array if empty

    first_entry = not summarized_data["results"]  # Check if file already has data

    # Process first 3 entries with a progress bar
    for entry in tqdm(data.get("results", []), desc="Processing Contexts", unit="entry"):
        if entry["index"] in existing_indices:
            continue  # Skip already processed entries

        context = entry.get("context", "")
        summary = generate_summary(context)

        summarized_entry = {
            "index": entry.get("index", ""),
            "context": context,
            "summary": summary
        }

        # Write entry progressively
        if not first_entry:
            f.write(",\n")  # Add comma only if there's already data

        json.dump(summarized_entry, f, indent=4, ensure_ascii=False)
        first_entry = False  # After first entry, always add comma

    f.write("\n  ]\n}")  # Close JSON array

print(f"Summarized data saved progressively to {output_file}")