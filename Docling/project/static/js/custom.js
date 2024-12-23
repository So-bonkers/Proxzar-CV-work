document.addEventListener("DOMContentLoaded", () => {
    // Get references to the forms
    const ingestForm = document.getElementById("ingestForm");
    const extractForm = document.getElementById("extractForm");
    const htmlToJsonForm = document.getElementById("htmlToJsonForm"); // New form for HTML to JSON

    // Utility function to handle responses
    async function handleResponse(response, resultElement) {
        const contentType = response.headers.get("content-type");
        if (contentType && contentType.includes("application/json")) {
            const data = await response.json();
            if (data.error) {
                resultElement.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            } else {
                return data;
            }
        } else {
            const text = await response.text();
            resultElement.innerHTML = `
                <div class="alert alert-danger">
                    <p>Unexpected server response:</p>
                    <pre>${text}</pre>
                </div>
            `;
        }
    }

    // Ingest form submission
    ingestForm.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(this);
        const ingestResult = document.getElementById("ingestResult");

        // Show a loading spinner while processing
        ingestResult.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Processing...';

        try {
            const res = await fetch('/api/v1/ingest', { method: 'POST', body: formData });
            const data = await handleResponse(res, ingestResult);
            if (data) {
                ingestResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>Client ID: <strong>${data.client_id}</strong></p>
                        <p>Output Path: <code>${data.output_path}</code></p>
                    </div>
                `;
            }
        } catch (err) {
            ingestResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };

    // Extract form submission
    extractForm.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(this);
        const extractResult = document.getElementById("extractResult");

        // Show a loading spinner while processing
        extractResult.innerHTML = '<div class="spinner-border text-success" role="status"></div> Processing...';

        try {
            const res = await fetch('/api/v1/extract', { method: 'POST', body: formData });
            const data = await handleResponse(res, extractResult);
            if (data) {
                extractResult.innerHTML = `
                    <div class="alert alert-success">
                        ${data.message}
                        <ul>${(data.files || []).map(file => `<li>${file}</li>`).join('')}</ul>
                    </div>
                `;
            }
        } catch (err) {
            extractResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };

    // HTML to JSON form submission
    htmlToJsonForm.onsubmit = async function (e) {
        e.preventDefault();
        const client_id_html = document.getElementById("client_id_html").value;
        const htmlToJsonResult = document.getElementById("htmlToJsonResult");

        // Show a loading spinner while processing
        htmlToJsonResult.innerHTML = '<div class="spinner-border text-warning" role="status"></div> Converting HTML to JSON...';

        try {
            const res = await fetch('/api/v1/htmlToJson', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_id: client_id_html })
            });
            const data = await handleResponse(res, htmlToJsonResult);
            if (data) {
                htmlToJsonResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>${data.message}</p>
                        <p>JSON file saved at: <code>${data.json_file}</code></p>
                    </div>
                `;
            }
        } catch (err) {
            htmlToJsonResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };
});
