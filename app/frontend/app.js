const askBtn = document.getElementById('askBtn');
const question = document.getElementById('question');
const solution = document.getElementById('solution');
const answer = document.getElementById('answer');
const errorEl = document.getElementById('error');
const statusEl = document.getElementById('status');
const sources = document.getElementById('sources');

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
