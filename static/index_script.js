document.addEventListener('DOMContentLoaded', function () {
    // Array to store questions and answers
    const questions = [
        {% for question in questions %}
            {
                name: "{{ question['name'] }}",
                message: "{{ question['message'] }}"
            },
        {% endfor %}
    ];

    // Variables to keep track of the current question index
    let currentQuestionIndex = 0;
    const questionContainer = document.getElementById('question-container');
    const nextQuestionBtn = document.getElementById('next-question-btn');

    // Function to display the current question
    function displayQuestion() {
        const currentQuestion = questions[currentQuestionIndex];
        questionContainer.innerHTML = `
            <label>${currentQuestion.message}</label>
            <select name="${currentQuestion.name}">
                <option value="yes">Yes</option>
                <option value="no">No</option>
            </select>
        `;
    }

    // Function to handle the "Next Question" button click
    function nextQuestion() {
        // Increase the question index
        currentQuestionIndex++;

        // If there are more questions, display the next question; otherwise, submit the form
        if (currentQuestionIndex < questions.length) {
            displayQuestion();
        } else {
            // Submit the form (you may want to implement the form submission logic here)
            alert('All questions answered. Submitting form...');
        }
    }

    // Initial display of the first question
    displayQuestion();

    // Event listener for the "Next Question" button click
    nextQuestionBtn.addEventListener('click', nextQuestion);
});
