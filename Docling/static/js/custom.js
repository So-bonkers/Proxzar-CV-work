document.addEventListener("DOMContentLoaded", () => {
    // Get references to the forms and containers
    const ingestPathForm = document.getElementById("ingestPathForm");
    const ingestLinkForm = document.getElementById("ingestLinkForm");
    const ingestStreamForm = document.getElementById("ingestStreamForm"); // New form reference
    const ingestResult = document.getElementById("ingestResult");
    const extractForm = document.getElementById("extractForm");
    const htmlToJsonForm = document.getElementById("htmlToJsonForm");

    // Utility function to handle API responses
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

    // Ingest via Path
    ingestPathForm.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(e.target);
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

    // Ingest via Link
    ingestLinkForm.onsubmit = async function (e) {
        e.preventDefault();
        const fileLink = document.getElementById("fileLink").value.trim();

        // Validate fileLink
        if (!fileLink) {
            alert("Please enter a valid file link!");
            return;
        }

        const urlPattern = /^(https?:\/\/)[^\s/$.?#].[^\s]*$/;
        if (!urlPattern.test(fileLink)) {
            alert("Invalid URL format! Please enter a valid link.");
            return;
        }

        ingestResult.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Processing...';

        try {
            const res = await fetch('/api/v1/ingest-link', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ file_link: fileLink })
            });
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

    // NEW: Ingest via Stream
    ingestStreamForm.onsubmit = async function (e) {
        e.preventDefault();
        const bucketName = document.getElementById("bucketName").value.trim();
        const fileKey = document.getElementById("fileKey").value.trim();

        // Validate inputs
        if (!bucketName || !fileKey) {
            alert("Please enter both bucket name and file key!");
            return;
        }

        ingestResult.innerHTML = '<div class="spinner-border text-info" role="status"></div> Processing stream...';

        try {
            const res = await fetch('/api/v1/ingest-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    bucket_name: bucketName,
                    file_key: fileKey
                })
            });
            const data = await handleResponse(res, ingestResult);
            if (data) {
                ingestResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>Client ID: <strong>${data.client_id}</strong></p>
                        <p>Output Path: <code>${data.output_path}</code></p>
                        <p>${data.message}</p>
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
        const formData = new FormData(e.target);
        const extractResult = document.getElementById("extractResult");

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