function goToPage(num) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('page' + num).classList.add('active');
}

document.addEventListener("DOMContentLoaded", function() {
  const uploadForm = document.getElementById("uploadForm");
  if (uploadForm) {
    uploadForm.addEventListener("submit", async function(event) {
      event.preventDefault();
      
      console.log("Form submission started");
      
      // إظهار مؤشر التحميل
      const submitButton = this.querySelector('button[type="submit"]');
      const originalText = submitButton.textContent;
      submitButton.textContent = 'جاري المعالجة...';
      submitButton.disabled = true;

      try {
        const formData = new FormData(this);
        let response = await fetch("/upload", {
          method: "POST",
          body: formData
        });

        console.log("Response received", response);

        let result = await response.json();
        console.log("Result parsed", result);

        if (result.error) {
          alert("Error: " + result.error);
          return;
        }

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
          if(row.predicted_fraud == 1 || row.predicted_fraud == "Fraud"){
            tr.style.backgroundColor = "#f28c8c"; // أحمر فاتح
          }
          tbody.appendChild(tr);
        });

        // Summary info
        document.getElementById("summary").innerText =
          `Detected ${result.fraud_count} fraudulent transactions out of ${result.total_count}`;

        goToPage(2);
      } catch (error) {
        console.error("Error:", error);
        alert("حدث خطأ أثناء المعالجة");
      } finally {
        // إعادة حالة الزر إلى الطبيعي
        submitButton.textContent = originalText;
        submitButton.disabled = false;
      }
    });
  }
});