document.addEventListener("DOMContentLoaded", async () => {
    const processForm = document.getElementById("processForm");
    const processResult = document.getElementById("processResult");
    const logOutput = document.getElementById("logOutput");

    async function getToken() {
        try {
            const res = await fetch('/api/v1/authenticate', { method: 'POST' });
            const data = await res.json();
            return data.access_token;
        } catch (err) {
            console.error("Error fetching token:", err);
            return null;
        }
    }

    function logMessage(message) {
        const logEntry = document.createElement("p");
        logEntry.classList.add("log-entry");
        logEntry.textContent = message;
        logOutput.prepend(logEntry);
        logOutput.scrollTop = 0;
    }

    processForm.onsubmit = async function (e) {
        e.preventDefault();
        const fileLink = document.getElementById("fileLink").value.trim();

        if (!fileLink) {
            alert("Please enter a valid file link!");
            return;
        }

        processResult.innerHTML = '<div class="loader">Processing...</div>';
        logMessage("🟡 Processing started...");

        const token = await getToken();
        if (!token) {
            processResult.innerHTML = `<div class="error">Error: Authentication failed</div>`;
            return;
        }

        try {
            const res = await fetch('/api/v1/process-file', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({ file_link: fileLink })
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let responseText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                responseText += chunk;
                logMessage(chunk);
            }

            // ✅ Fix: Parse JSON from the accumulated response text
            const data = JSON.parse(responseText.trim());

            if (data.error) {
                processResult.innerHTML = `<div class="error">${data.error}</div>`;
                logMessage(`🔴 Error: ${data.error}`);
            } else {
                processResult.innerHTML = `<div class="success">
                    <p>✅ File processed successfully!</p>
                    <p>🔹 Client ID: <b>${data.client_id}</b></p>
                    <p>📄 JSON Output: <a href="${data.output_json}" target="_blank">View JSON</a></p>
                </div>`;
                logMessage("🟢 Processing completed successfully!");
            }
        } catch (err) {
            processResult.innerHTML = `<div class="error">Error: ${err.message}</div>`;
            logMessage(`🔴 Error: ${err.message}`);
        }
    };
});
