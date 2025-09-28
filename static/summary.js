let summaryData = null;
let summaryChart = null;

const dropZoneSummary = document.getElementById('dropZoneSummary');
dropZoneSummary.addEventListener('click', () => document.getElementById('fileInputSummary').click());
dropZoneSummary.addEventListener('dragover', e => { e.preventDefault(); dropZoneSummary.style.background='#e8f4fc'; });
dropZoneSummary.addEventListener('dragleave', e => { e.preventDefault(); dropZoneSummary.style.background='#f8f9fa'; });
dropZoneSummary.addEventListener('drop', e => {
    e.preventDefault();
    dropZoneSummary.style.background = '#f8f9fa';
    if(e.dataTransfer.files.length > 0) handleFileSelectionSummary(e.dataTransfer.files[0]);
});

document.getElementById('fileInputSummary').addEventListener('change', function(){ handleFileSelectionSummary(this.files[0]); });

function handleFileSelectionSummary(file){
    const fileSize = file.size<1024*1024 ? (file.size/1024).toFixed(2)+' KB' : (file.size/(1024*1024)).toFixed(2)+' MB';
    document.getElementById('fileNameSummary').textContent = file.name;
    document.getElementById('fileSizeSummary').textContent = ' ('+fileSize+')';
    document.getElementById('fileInfoSummary').style.display='flex';
    hideAlertsSummary();
}

function processSummaryFile(){
    const fileInput = document.getElementById('fileInputSummary');
    if(!fileInput.files.length){ showErrorSummary("Please select a file first"); return; }
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('file', file);
    const btn = document.querySelector('#fileInfoSummary .btn-success');
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...'; btn.disabled=true;
    hideAlertsSummary();

    fetch('/get_summary', { method:'POST', body:formData })
    .then(res=>res.json())
    .then(data=>{ if(data.error) throw new Error(data.error); summaryData=data; displaySummary(data); showSuccessSummary("Summary generated successfully!"); })
    .catch(err=>showErrorSummary("Error: "+err.message))
    .finally(()=>{ btn.innerHTML='<i class="fas fa-play"></i> Generate Summary'; btn.disabled=false; });
}

function displaySummary(data){
    const container = document.getElementById('summaryStats'); container.innerHTML=''; container.style.display='flex';
    const cards = [ {label:'Rows', value:data.rows}, {label:'Columns', value:data.columns}, {label:'Missing Values', value:data.missing} ];
    cards.forEach(c=>{
        const card=document.createElement('div'); card.classList.add('stat-card');
        card.innerHTML=`<div class="stat-value">${c.value}</div><div class="stat-label">${c.label}</div>`;
        container.appendChild(card);
    });

    if(data.numeric && data.numeric.length>0){
        const chartContainer=document.getElementById('summaryChartContainer'); chartContainer.style.display='block';
        const ctx=document.getElementById('summaryChart').getContext('2d');
        if(summaryChart) summaryChart.destroy();
        summaryChart = new Chart(ctx, {
            type:'bar',
            data:{ labels:data.numeric[0].values.map((_,i)=>i+1), datasets:[{label:data.numeric[0].column, data:data.numeric[0].values, backgroundColor:'#3498db'}] },
            options:{ responsive:true, plugins:{ legend:{ display:false } } }
        });
    }
}

function showErrorSummary(msg){ const a=document.getElementById('errorAlertSummary'); a.textContent=msg; a.style.display='block'; setTimeout(()=>a.style.display='none',5000); }
function showSuccessSummary(msg){ const a=document.getElementById('successAlertSummary'); a.textContent=msg; a.style.display='block'; setTimeout(()=>a.style.display='none',3000); }
function hideAlertsSummary(){ document.getElementById('errorAlertSummary').style.display='none'; document.getElementById('successAlertSummary').style.display='none'; }
