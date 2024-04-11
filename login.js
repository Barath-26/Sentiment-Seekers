// Function to go back
function goBack() {
    window.history.back();
}

// Additional JavaScript code (if needed) can be added below this line


// Event listener for the form submission
document.getElementById("loginForm").addEventListener("submit", function (event) {
    event.preventDefault();

    // Replace the following line with your login logic
    alert("Login logic will go here!");

    // For demonstration purposes, let's navigate to a hypothetical home page
    window.location.href = "hackathon.html";
});
