document.addEventListener("DOMContentLoaded", () => {
    const ingestForm = document.getElementById("ingestForm");
    const extractForm = document.getElementById("extractForm");

    ingestForm.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(this);
        const ingestResult = document.getElementById("ingestResult");

        ingestResult.innerHTML = '<div class="spinner-border text-primary" role="status"></div> Processing...';

        try {
            const res = await fetch('/api/v1/ingest', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.client_id) {
                ingestResult.innerHTML = `
                    <div class="alert alert-success">
                        <p>Client ID: <strong>${data.client_id}</strong></p>
                        <p>Output Path: <code>${data.output_path}</code></p>
                    </div>
                `;
            } else {
                ingestResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (err) {
            ingestResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };

    extractForm.onsubmit = async function (e) {
        e.preventDefault();
        const formData = new FormData(this);
        const extractResult = document.getElementById("extractResult");

        extractResult.innerHTML = '<div class="spinner-border text-success" role="status"></div> Processing...';

        try {
            const res = await fetch('/api/v1/extract', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.message) {
                extractResult.innerHTML = `
                    <div class="alert alert-success">${data.message}</div>
                `;
            } else {
                extractResult.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
            }
        } catch (err) {
            extractResult.innerHTML = `<div class="alert alert-danger">Error: ${err.message}</div>`;
        }
    };
});
