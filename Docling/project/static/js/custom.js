document.addEventListener("DOMContentLoaded", () => {
    // Get references to the forms
    const ingestForm = document.getElementById("ingestForm");
    const extractForm = document.getElementById("extractForm");
    const htmlToJsonForm = document.getElementById("htmlToJsonForm"); // Form for HTML to JSON

    // Containers for switching functionality in ingestion
    const ingestResult = document.getElementById("ingestResult");
    let ingestMode = null; // To track which ingest method is selected

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

    // Ingest button handlers
    document.getElementById("ingestViaPathBtn").addEventListener("click", () => {
        ingestMode = "path";
        ingestResult.innerHTML = `
            <form id="pathIngestForm" enctype="multipart/form-data">
                <div class="mb-3">
                    <label for="file" class="form-label">Choose File</label>
                    <input
                        type="file"
                        class="form-control"
                        id="file"
                        name="file"
                        required
                    />
                </div>
                <button type="submit" class="btn btn-primary">Ingest via Path</button>
            </form>
        `;

        const pathIngestForm = document.getElementById("pathIngestForm");
        pathIngestForm.onsubmit = handleIngestViaPath;
    });

    document.getElementById("ingestViaLinkBtn").addEventListener("click", () => {
        ingestMode = "link";
        ingestResult.innerHTML = `
            <form id="linkIngestForm">
                <div class="mb-3">
                    <label for="fileLink" class="form-label">Enter File Link</label>
                    <input
                        type="url"
                        class="form-control"
                        id="fileLink"
                        name="fileLink"
                        placeholder="https://example.com/document.pdf"
                        required
                    />
                </div>
                <button type="submit" class="btn btn-primary">Ingest via Link</button>
            </form>
        `;

        const linkIngestForm = document.getElementById("linkIngestForm");
        linkIngestForm.onsubmit = handleIngestViaLink;
    });

    // Ingest via Path
    async function handleIngestViaPath(e) {
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
    }

    // Ingest via Link
    async function handleIngestViaLink(e) {
        e.preventDefault();
        const fileLink = document.getElementById("fileLink").value;
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
    }

    // Other forms' submission handlers (Extract, HTML to JSON) remain unchanged...
});
