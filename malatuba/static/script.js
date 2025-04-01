function sendMessage(event) {
    if (event && event.key !== "Enter") return;

    let inputField = document.getElementById("userInput");
    let userMessage = inputField.value.trim();
    if (userMessage === "") return;

    let apiChatbox = document.getElementById("apiMessages");
    let geminiChatbox = document.getElementById("geminiMessages");

    // Display user message in both chat windows
    let userDiv1 = document.createElement("div");
    userDiv1.textContent = "You: " + userMessage;
    userDiv1.style.fontWeight = "bold";
    apiChatbox.appendChild(userDiv1);

    let userDiv2 = document.createElement("div");
    userDiv2.textContent = "You: " + userMessage;
    userDiv2.style.fontWeight = "bold";
    geminiChatbox.appendChild(userDiv2);

    // Fetch diagnosis from API
    fetch(`/diagnosis?symptoms=${userMessage}`)
        .then(response => response.json())
        .then(data => {
            let botDiv = document.createElement("div");
            if (data.error) {
                botDiv.textContent = "Bot: " + data.error;
            } else if (data.message) {
                botDiv.textContent = "Bot: " + data.message;
            } else {
                botDiv.innerHTML = "Bot: Possible diseases found:<br>";
                data.forEach(disease => {
                    botDiv.innerHTML += `<strong>${disease.name}</strong> (ICD-10: ${disease.icd10_codes}) <br>
                    <a href="${disease.info_link}" target="_blank">More info</a><br><br>`;
                });
            }
            apiChatbox.appendChild(botDiv);
            apiChatbox.scrollTop = apiChatbox.scrollHeight;
        });

    // Fetch response from Gemini AI
    fetch('/gemini', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMessage })
    })
    .then(response => response.json())
    .then(data => {
        let geminiDiv = document.createElement("div");
        geminiDiv.textContent = "Gemini: " + (data.response || "No response");
        geminiChatbox.appendChild(geminiDiv);
        geminiChatbox.scrollTop = geminiChatbox.scrollHeight;
    });

    inputField.value = "";
}
