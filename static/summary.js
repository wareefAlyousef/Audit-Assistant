document.addEventListener("DOMContentLoaded", () => {
  fetch("/get_summary")
    .then(response => response.json())
    .then(data => {
      const summaryElement = document.getElementById("summary");
      if (data.summary) {
        summaryElement.textContent = data.summary;
      } else {
        summaryElement.textContent = "Error loading summary.";
      }
    })
    .catch(error => {
      console.error("Error fetching summary:", error);
      document.getElementById("summary").textContent = "Failed to fetch summary.";
    });
});