async function summarizeText() {
    const textInput = document.getElementById("textInput").value;
    const resultsDiv = document.getElementById("results");
    const loadingDiv = document.getElementById("loading");
    const progressContainer = document.getElementById("progress-container");
    const progressBar = document.getElementById("progress-fill");
    const progressText = document.getElementById("progress-text");
    const totalText = document.getElementById("total-text");

    resultsDiv.innerHTML = "";  
    progressContainer.classList.add("hidden");

    if (!textInput.trim()) {
        alert("Please enter some text to summarize.");
        return;
    }

    const paragraphs = textInput.split("\n").filter(p => p.trim() !== ""); 
    const apiUrl = "/summarize"; 

    loadingDiv.classList.remove("hidden"); 
    progressContainer.classList.remove("hidden"); 
    progressText.innerText = "0"; 
    totalText.innerText = paragraphs.length;
    progressBar.style.width = "0%"; 

    try {
        const response = await fetch(apiUrl, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ paragraphs })
        });

        const data = await response.json();
        loadingDiv.classList.add("hidden");

        if (data.summaries) {
            data.stored_data.forEach((entry, index) => {
                const summaryBox = document.createElement("div");
                summaryBox.classList.add("summary-box");
                summaryBox.innerHTML = `<strong>Summary ${entry.index}:</strong> ${entry.summary}`;
                resultsDiv.appendChild(summaryBox);

                progressText.innerText = index + 1;
                progressBar.style.width = `${((index + 1) / paragraphs.length) * 100}%`;
            });
        } else {
            resultsDiv.innerHTML = `<p style="color: red;">Error: ${data.error}</p>`;
        }
    } catch (error) {
        loadingDiv.classList.add("hidden"); 
        resultsDiv.innerHTML = `<p style="color: red;">Failed to connect to API.</p>`;
    }
}

async function clearSession() {
    const apiUrl = "/clear_session";

    try {
        const response = await fetch(apiUrl, {
            method: "POST"
        });

        const data = await response.json();
        alert(data.message);
        document.getElementById("results").innerHTML = "";
    } catch (error) {
        alert("Failed to clear session.");
    }
}
