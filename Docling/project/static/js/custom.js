document.addEventListener("DOMContentLoaded", () => {
    // Get references to the forms
    const ingestForm = document.getElementById("ingestForm");
    const extractForm = document.getElementById("extractForm");
    const htmlToJsonForm = document.getElementById("htmlToJsonForm"); // New form for HTML to JSON

    // Ingest form submission
    ingestForm.onsubmit = async function (e) {
        e.preventDefault(); // Prevent default form submission
        const formData = new FormData(this); // Collect form data
        const ingestResult = document.getElementById("ingestResult");

        // Show a loading spinner while processing
        ingestResult.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Processing...';

        try {
            // Send form data to the server
            const res = await fetch('/api/v1/ingest', { method: 'POST', body: formData });
            const data = await res.json(); // Parse the JSON response
            if (data.client_id) {
                // Display success message with client ID and output path
                ingestResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>Client ID: <strong>${data.client_id}</strong></p>
                        <p>Output Path: <code>${data.output_path}</code></p>
                    </div>
                `;
            } else {
                // Display error message
                ingestResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (err) {
            // Display error message if the request fails
            ingestResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };

    // Extract form submission
    extractForm.onsubmit = async function (e) {
        e.preventDefault(); // Prevent default form submission
        const formData = new FormData(this); // Collect form data
        const extractResult = document.getElementById("extractResult");

        // Show a loading spinner while processing
        extractResult.innerHTML = '<div class="spinner-border text-success" role="status"></div> Processing...';

        try {
            // Send form data to the server
            const res = await fetch('/api/v1/extract', { method: 'POST', body: formData });
            const data = await res.json(); // Parse the JSON response
            if (data.message) {
                // Display success message
                extractResult.innerHTML = `
                    <div class="alert alert-success">${data.message}</div>
                `;
            } else {
                // Display error message
                extractResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (err) {
            // Display error message if the request fails
            extractResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };

    // New HTML to JSON form submission
    htmlToJsonForm.onsubmit = async function (e) {
        e.preventDefault(); // Prevent default form submission
        const client_id_html = document.getElementById("client_id_html").value; // Get client ID from input
        const htmlToJsonResult = document.getElementById("htmlToJsonResult");

        // Show a loading spinner while processing
        htmlToJsonResult.innerHTML = '<div class="spinner-border text-warning" role="status"></div> Converting HTML to JSON...';

        try {
            // Send client ID to the server
            const res = await fetch('/api/v1/html-to-json', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ client_id: client_id_html })
            });
            const data = await res.json(); // Parse the JSON response
            if (data.message) {
                // Display success message with JSON file path
                htmlToJsonResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>${data.message}</p>
                        <p>JSON file saved at: <code>${data.json_file}</code></p>
                    </div>
                `;
            } else {
                // Display error message
                htmlToJsonResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (err) {
            // Display error message if the request fails
            htmlToJsonResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };
});
