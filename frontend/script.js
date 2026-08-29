async function askQuestion() {

    const questionInput = document.getElementById("question");
    const answerBox = document.getElementById("answer");
    const loading = document.getElementById("loading");

    const query = questionInput.value.trim();

    if (!query) {
        return;
    }

    loading.classList.remove("hidden");

    answerBox.innerHTML = "";

    try {

        const response = await fetch("/chat/", {
            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                query: query
            })
        });

        if (!response.ok) {
            throw new Error("Request failed");
        }

        const data = await response.json();

        answerBox.textContent = data.answer;

    } catch (error) {

        answerBox.textContent =
            "Something went wrong. Make sure the FastAPI server is running.";

        console.error(error);

    } finally {

        loading.classList.add("hidden");

    }
}