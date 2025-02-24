import json
import torch
import logging
import os
from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = "supersecretkey"  # Needed for session management
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define JSON storage file
SESSION_FILE = "session_summaries.json"

# Load fine-tuned Pegasus model and tokenizer
MODEL_PATH = "pegasus-finetuned"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Loading model on {device}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

# Serve index.html at root URL
@app.route("/")
def home():
    """
    Serve the index.html file at the root URL.

    Returns:
        str: Rendered HTML template for the index page.
    """
    return render_template("index.html")

# Function to generate summaries
def generate_summary(paragraph):
    """
    Generate a summary for a given paragraph using the Pegasus model.

    Args:
        paragraph (str): The input paragraph to summarize.

    Returns:
        str: The generated summary or an error message if input is invalid.
    """
    if not paragraph.strip():
        return "Invalid input: Empty text received."

    try:
        # Tokenize the input paragraph and move tensors to the appropriate device
        inputs = tokenizer(paragraph, return_tensors="pt", max_length=1024, truncation=True).to(device)
        
        # Generate summary IDs using the model
        with torch.no_grad():
            summary_ids = model.generate(inputs["input_ids"], max_length=256, num_beams=5, early_stopping=True)
        
        # Decode the summary IDs to get the summary text
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return "Error: Unable to generate summary."

# Function to save session data to JSON
def save_session_to_json():
    """
    Save the current session data to a JSON file.

    This function writes the session data to a file named `session_summaries.json`.
    """
    try:
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session.get("session_data", []), f, indent=4, ensure_ascii=False)
        logger.info("Session data saved to session_summaries.json")
    except Exception as e:
        logger.error(f"Failed to save session data: {str(e)}")

# API endpoint to summarize and store responses in session + JSON
@app.route("/summarize", methods=["POST"])
def summarize():
    """
    API endpoint to summarize multiple paragraphs and store responses in session and JSON file.

    Expects a JSON payload with a "paragraphs" key containing a list of paragraphs.

    Returns:
        JSON: A JSON response containing the summaries and stored session data or an error message.
    """
    try:
        # Parse the JSON request data
        data = request.get_json()
        paragraphs = data.get("paragraphs", [])

        # Validate the request format
        if not isinstance(paragraphs, list) or not all(isinstance(p, str) for p in paragraphs):
            return jsonify({"error": "Invalid request format. Expected a list of text paragraphs."}), 400

        # Check if paragraphs are provided
        if not paragraphs:
            return jsonify({"error": "No paragraphs provided for summarization."}), 400

        # Generate summaries for each paragraph
        summaries = [generate_summary(p) for p in paragraphs]

        # Store responses temporarily in the session
        if "session_data" not in session:
            session["session_data"] = []

        for i, summary in enumerate(summaries):
            session["session_data"].append({
                "index": len(session["session_data"]) + 1,
                "context": paragraphs[i],
                "summary": summary
            })

        session.modified = True  # Ensure session updates persist
        save_session_to_json()  # Save session data to JSON file

        return jsonify({"summaries": summaries, "stored_data": session["session_data"]})

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

# API endpoint to clear session data (and delete JSON file)
@app.route("/clear_session", methods=["POST"])
def clear_session():
    """
    API endpoint to clear session data and delete the JSON file.

    Returns:
        JSON: A JSON response indicating the session data has been cleared.
    """
    session.clear()
    if os.path.exists(SESSION_FILE):
        os.remove(SESSION_FILE)
        logger.info("Session file deleted.")
    return jsonify({"message": "Session data cleared."})

# Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
