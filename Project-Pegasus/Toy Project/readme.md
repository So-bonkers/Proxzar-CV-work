# Ticket # 949523

## Create a Flask api endpoint for PEGASUSX

### Objectives:
1. Finetune the below PEGASUS-X model with few shot examples - summaries generated from our DeepSeek model inference endpoint from the construction corpus file named QG_custom_range.json_1.json
    1. Model name: google/pegasus-x-base
    2. Model doc: https://huggingface.co/docs/transformers/en/model_doc/pegasus_x
2. Build a Flask API endpoint for generating summaries for multiple paragraphs using the above fine-tuned model.

## File Structure:
```graphql
/project_directory
│── flaskAPI.py                  # Flask API for text summarization
│── pegasus-finetuned            # Contains the finetuned model
│── session_summaries.json        # Stores session-based summaries (automatically-generated-during-session)
│── templates/
│   ├── index.html                # Web interface for testing summarization
│── static/
│   ├── styles.css                # UI styling
│   ├── script.js                 # Handles API calls and UI updates
│── requirements.txt              # you need to pip install this file before the first run
│── README.md                     # This documentation file
```

##  Requirements

- Python 3.8+
- Pip
- Virtual Environment (optional but recommended)
- CUDA-enabled GPU (optional for better performance)

## Start Flask API
To run the API:

```bash
python flaskAPI.py
```
The API will be available at:

```cpp
http://127.0.0.1:5000/
```

# Web Interface Usage

- Open a brows$er and go to $ http://127.0.0.1:5000/$
- Enter multiple paragraphs of text in the input box.
- Click the "Summarize" button to generate concise summaries.
- The progress bar will update as summaries are processed.
- Summaries are displayed below, stored in session_summaries.json.
- To clear session data, click the "Refresh Session" button.

## API Endpoints

| **Method** | **Endpoint**       | **Description** |
|------------|-------------------|----------------|
| `GET`      | `/`               | Serves the Web UI |
| `POST`     | `/summarize`      | Generates summaries for input text |
| `POST`     | `/clear_session`  | Clears stored session summaries |


## Example API call

To summarize multiple paragraphs using ```cURL```:

```bash
curl -X POST "http://127.0.0.1:5000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"paragraphs":["The construction industry is growing rapidly.","AI is being used for automation in construction."]}'
```