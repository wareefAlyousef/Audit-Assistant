function goToPage(num) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page' + num).classList.add('active');
}

// Handle file upload
document.getElementById("uploadForm").addEventListener("submit", async function(event) {
  event.preventDefault();

  const formData = new FormData(this);

  let response = await fetch("/upload", {
    method: "POST",
    body: formData
  });

  let result = await response.json();

  // Populate results table
  let tbody = document.querySelector("#resultsTable tbody");
  tbody.innerHTML = "";

  result.data.forEach(row => {
    let tr = document.createElement("tr");
    Object.values(row).forEach(val => {
      let td = document.createElement("td");
      td.textContent = val;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });

  // Summary info
  document.getElementById("summary").innerText =
    `Detected ${result.fraud_count} fraudulent transactions out of ${result.total_count}`;

  goToPage(2);
});