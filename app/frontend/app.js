const askBtn = document.getElementById('askBtn');
const question = document.getElementById('question');
const solution = document.getElementById('solution');
const answer = document.getElementById('answer');
const errorEl = document.getElementById('error');
const statusEl = document.getElementById('status');
const sources = document.getElementById('sources');
const ocrBtn = document.getElementById('ocrBtn');
const ocrFile = document.getElementById('ocrFile');
const ocrText = document.getElementById('ocrText');
const cfgBtn = document.getElementById('cfgBtn');
const cfgText = document.getElementById('cfgText');

function setStatus(msg) {
  statusEl.textContent = msg;
}

async function ask() {
  const payload = {
    query: question.value.trim(),
    solution: solution.value.trim(),
    language: document.getElementById('language').value,
    show_steps: document.getElementById('showSteps').checked,
    mode: document.getElementById('mode').value,
    top_k: parseInt(document.getElementById('topK').value, 10) || 5,
  };

  if (!payload.query) {
    setStatus('Please enter a question.');
    return;
  }

  setStatus('Thinking...');
  answer.textContent = '';
  errorEl.textContent = '';
  sources.innerHTML = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error('Request failed');
    const data = await res.json();

    answer.textContent = data.answer || '';
    if (data.error) errorEl.textContent = data.error;

    const list = document.createElement('ul');
    (data.sources || []).forEach((s) => {
      const li = document.createElement('li');
      const file = s.source.split('\\').pop().split('/').pop();
      li.textContent = `${file} ? page ${s.page}`;
      list.appendChild(li);
    });
    sources.appendChild(list);
    setStatus('');
  } catch (err) {
    setStatus('Error. Check backend logs.');
  }
}

askBtn.addEventListener('click', ask);

async function runOcr() {
  if (!ocrFile.files || !ocrFile.files[0]) {
    ocrText.textContent = 'Please choose an image.';
    return;
  }
  const form = new FormData();
  form.append('image', ocrFile.files[0]);
  ocrText.textContent = 'Reading...';
  try {
    const res = await fetch('/api/ocr', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'OCR failed');
    ocrText.textContent = data.text || '';
  } catch (err) {
    ocrText.textContent = `Error: ${err.message}`;
  }
}

ocrBtn.addEventListener('click', runOcr);

async function loadConfig() {
  cfgText.textContent = 'Loading...';
  try {
    const res = await fetch('/api/config');
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Config error');
    cfgText.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    cfgText.textContent = `Error: ${err.message}`;
  }
}

cfgBtn.addEventListener('click', loadConfig);
