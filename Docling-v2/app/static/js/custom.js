document.addEventListener("DOMContentLoaded", () => {
    const resultContainer = document.getElementById("result");

    // Function to get JWT token automatically
    async function getToken() {
        try {
            const response = await fetch("/api/v2/token", {
                method: "POST",
                headers: { "Content-Type": "application/x-www-form-urlencoded" },
                body: "username=testuser&password=password123"  // Change to your credentials
            });

            if (!response.ok) {
                throw new Error("Authentication failed");
            }

            const data = await response.json();
            return data.access_token;
        } catch (err) {
            resultContainer.innerHTML = `<div class="error">❌ Authentication Error: ${err.message}</div>`;
            return null;
        }
    }

    // Function to send API requests with token
    async function sendRequest(endpoint, body, isFormData = false) {
        const token = await getToken();
        if (!token) return; // Stop if authentication fails

        const headers = { "Authorization": `Bearer ${token}` };
        if (!isFormData) headers["Content-Type"] = "application/json";

        const res = await fetch(endpoint, {
            method: "POST",
            headers: headers,
            body: isFormData ? body : JSON.stringify(body),
        });

        const data = await res.json();
        handleResponse(data);
    }

    // Function to handle API responses
    function handleResponse(data) {
        if (data.error) {
            resultContainer.innerHTML = `<div class="error">❌ ${data.error}</div>`;
        } else {
            resultContainer.innerHTML = `<div class="success">✅ ${data.message}<br>📄 JSON Output: <a href="${data.output_json}" target="_blank">View JSON</a></div>`;
        }
    }

    // Convert by Link
    document.getElementById("convertLinkForm").onsubmit = async (e) => {
        e.preventDefault();
        resultContainer.innerHTML = "⏳ Processing...";

        const proxzarKeyID = document.getElementById("proxzarKeyIDLink").value;
        const fileLink = document.getElementById("fileLink").value;

        await sendRequest("/api/v2/convertLink", { proxzarKeyID, file_link: fileLink });
    };

    // Convert by Upload
    document.getElementById("convertUploadForm").onsubmit = async (e) => {
        e.preventDefault();
        resultContainer.innerHTML = "⏳ Uploading & Processing...";

        const formData = new FormData();
        formData.append("proxzarKeyID", document.getElementById("proxzarKeyIDUpload").value);
        formData.append("file", document.getElementById("fileUpload").files[0]);

        await sendRequest("/api/v2/convertUpload", formData, true);
    };

    // Convert by S3
    document.getElementById("convertS3Form").onsubmit = async (e) => {
        e.preventDefault();
        resultContainer.innerHTML = "⏳ Processing from S3...";

        const proxzarKeyID = document.getElementById("proxzarKeyIDS3").value;
        const bucketName = document.getElementById("bucketName").value;
        const fileKey = document.getElementById("fileKey").value;

        await sendRequest("/api/v2/convertS3", { proxzarKeyID, bucket_name: bucketName, file_key: fileKey });
    };
});
