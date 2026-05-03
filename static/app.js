// Handles frontend logic for uploading MRI files and showing results
// Front-end controller for the MRI ACR QA application.
// Comments mark the main UI update and API communication steps.

const state = {
  sets: [],
  currentIndex: -1,
  currentData: null,
  loading: false,
};

function statusClass(status) {
  return status === 'PASS' ? 'pass' : 'fail';
}

function setLoading(message) {
  state.loading = Boolean(message);
  document.getElementById('uploadStatus').textContent = message || '';
}

function clearDashboard() {
  document.getElementById('currentSetTitle').textContent = 'No set selected';
  document.getElementById('setStatusText').textContent = '';
  document.getElementById('passedCount').textContent = 'Passed: 0';
  document.getElementById('failedCount').textContent = 'Failed: 0';
  document.getElementById('datasetOverview').innerHTML = '<p class="muted">Upload a ZIP file to view dataset details.</p>';
  document.getElementById('moduleResults').innerHTML = '<p class="muted">Module results will appear after analysis.</p>';
  document.getElementById('downloadReportBtn').disabled = true;
}

function renderSetList() {
  const list = document.getElementById('setList');
  list.innerHTML = '';
  document.getElementById('setCount').textContent = String(state.sets.length);

  state.sets.forEach((setItem, index) => {
    const row = document.createElement('div');
    row.className = `set-row ${index === state.currentIndex ? 'active' : ''}`;

    const button = document.createElement('button');
    button.className = 'set-select';
    button.innerHTML = `<span>${index + 1}. ${setItem.name}</span>`;
    button.onclick = async () => {
      state.currentIndex = index;
      await loadAnalysis(setItem.name);
    };

    const actions = document.createElement('div');
    actions.className = 'set-actions';

    if (setItem.source === 'uploaded_sets') {
      const del = document.createElement('button');
      del.className = 'icon-btn danger';
      del.textContent = 'Delete';
      del.onclick = async (event) => {
        event.stopPropagation();
        if (!confirm(`Delete ${setItem.name}?`)) return;
        const response = await fetch(`/api/sets/${encodeURIComponent(setItem.name)}`, { method: 'DELETE' });
        const result = await response.json();
        if (!response.ok) {
          alert(result.error || 'Delete failed');
          return;
        }
        clearDashboard();
        await loadSets();
      };
      actions.appendChild(del);
    }

    row.appendChild(button);
    row.appendChild(actions);
    list.appendChild(row);
  });
}

function renderHeader() {
  const data = state.currentData;
  if (!data) return;
  document.getElementById('currentSetTitle').textContent = data.name;
  document.getElementById('setStatusText').textContent = data.validation;
  document.getElementById('passedCount').textContent = `Passed: ${data.passed}`;
  document.getElementById('failedCount').textContent = `Failed: ${data.failed}`;
  const reportBtn = document.getElementById('downloadReportBtn');
  reportBtn.disabled = false;
  reportBtn.onclick = () => window.open(data.report_url, '_blank');
}

function buildSliceList(item) {
  if (!item.available) {
    return `<p class="muted">Series not found.</p>`;
  }
  const firstImage = item.slices[0]?.image_url || '';
  const listHtml = item.slices.map(slice => `
    <button class="slice-btn" data-image="${slice.image_url}" data-name="Slice ${slice.number} - ${slice.filename}">
      Slice ${slice.number}
    </button>
  `).join('');

  return `
    <div class="slice-meta">${item.metadata.description || 'No description'}</div>
    <details class="slice-details">
      <summary>Slice List (${item.count})</summary>
      <div class="slice-layout">
        <div class="slice-buttons">${listHtml}</div>
        <div class="slice-preview">
          <img src="${firstImage}" alt="slice preview" class="slice-image" />
          <div class="slice-caption">Slice 1</div>
        </div>
      </div>
    </details>
  `;
}

function wireSliceButtons() {
  document.querySelectorAll('.series-slices, .slice-details').forEach(details => {
    const preview = details.querySelector('.slice-preview');
    details.querySelectorAll('.slice-chip, .slice-btn').forEach(btn => {
      btn.onclick = () => {
        if (!preview) return;
        let img = preview.querySelector('.slice-image');
        if (!img) {
          preview.innerHTML = '<img alt="slice preview" class="slice-image" /><div class="slice-caption"></div>';
          img = preview.querySelector('.slice-image');
        }
        const cap = preview.querySelector('.slice-caption');
        img.src = btn.dataset.image;
        if (cap) cap.textContent = btn.dataset.name;
      };
    });
  });
}

function metaLine(item) {
  const m = item.metadata || {};
  const parts = [];
  if (m.tr !== null && m.tr !== undefined) parts.push(`TR ${Number(m.tr).toFixed(2)}`);
  if (m.te !== null && m.te !== undefined) parts.push(`TE ${Number(m.te).toFixed(2)}`);
  if (m.thickness !== null && m.thickness !== undefined) parts.push(`Thickness ${Number(m.thickness).toFixed(2)} mm`);
  if (m.gap !== null && m.gap !== undefined) parts.push(`Gap ${Number(m.gap).toFixed(2)} mm`);
  return parts.length ? `<div class="series-meta-line">${parts.join(' • ')}</div>` : '';
}

function renderOverview() {
  const box = document.getElementById('datasetOverview');
  box.innerHTML = '';
  if (!state.currentData) return;

  const seriesList = state.currentData.all_series || Object.values(state.currentData.overview || {});
  if (!seriesList.length) {
    box.innerHTML = '<p class="muted">No valid MR series found in this upload.</p>';
    return;
  }

  const rows = seriesList.map((item, index) => {
    const m = item.metadata || {};
    const tr = m.tr !== null && m.tr !== undefined ? Number(m.tr).toFixed(2) : '—';
    const te = m.te !== null && m.te !== undefined ? Number(m.te).toFixed(2) : '—';
    const th = m.thickness !== null && m.thickness !== undefined ? `${Number(m.thickness).toFixed(2)} mm` : '—';
    const gap = m.gap !== null && m.gap !== undefined ? `${Number(m.gap).toFixed(2)} mm` : '—';
    const description = m.description || item.description || 'No description';
    const sliceButtons = item.available && item.slices
      ? item.slices.map(slice => `<button class="slice-chip" data-image="${slice.image_url}" data-name="Slice ${slice.number} - ${slice.filename}">S${slice.number}</button>`).join('')
      : '';

    return `
      <div class="series-list-row">
        <div class="series-title">
          <strong>${index + 1}. ${item.title}</strong>
          <span>${description}</span>
        </div>
        <div class="series-mini"><b>${item.available ? item.count : '—'}</b><small>files</small></div>
        <div class="series-mini"><b>${tr}</b><small>TR</small></div>
        <div class="series-mini"><b>${te}</b><small>TE</small></div>
        <div class="series-mini"><b>${th}</b><small>thick</small></div>
        <div class="series-mini"><b>${gap}</b><small>gap</small></div>
        <details class="series-slices lazy-slices">
          <summary>View slices (${item.count || 0})</summary>
          <div class="slice-compact-layout">
            <div class="slice-chip-list">${sliceButtons}</div>
            <div class="slice-preview compact empty-preview"><span>Select a slice to preview</span></div>
          </div>
        </details>
      </div>
    `;
  }).join('');

  box.innerHTML = `<div class="series-list">${rows}</div>`;
  wireSliceButtons();
}

function renderModules() {
  const box = document.getElementById('moduleResults');
  box.innerHTML = '';
  if (!state.currentData) return;

  state.currentData.modules.forEach(module => {
    const rows = module.measurements.length
      ? module.measurements.map(item => `
          <div class="measure-row">
            <strong>${item.label}</strong>
            <span>${item.value}</span>
            <span>${item.target}</span>
            <span class="tag ${statusClass(item.result)}">${item.result}</span>
            <small>${item.why}</small>
          </div>
        `).join('')
      : `<p class="muted">No measurements available.</p>`;

    const card = document.createElement('div');
    card.className = 'module-card';
    card.innerHTML = `
      <div class="module-head">
        <h3>${module.name}</h3>
        <span class="tag ${statusClass(module.status)}">${module.status}</span>
      </div>
      <p class="muted">${module.why}</p>
      <div class="measure-table">
        <div class="measure-header">
          <span>Check</span><span>Measurement</span><span>Target</span><span>Result</span><span>Why</span>
        </div>
        ${rows}
      </div>
    `;
    box.appendChild(card);
  });
}

async function loadSets(preferredName = null) {
  const response = await fetch('/api/sets');
  const result = await response.json();
  state.sets = result;
  if (!state.sets.length) {
    state.currentIndex = -1;
    state.currentData = null;
    renderSetList();
    clearDashboard();
    return;
  }
  const preferredIndex = preferredName ? state.sets.findIndex(item => item.name === preferredName) : -1;
  if (preferredIndex >= 0) {
    state.currentIndex = preferredIndex;
  } else if (state.currentIndex < 0 || state.currentIndex >= state.sets.length) {
    state.currentIndex = 0;
  }
  renderSetList();
  await loadAnalysis(state.sets[state.currentIndex].name);
}

async function loadAnalysis(name) {
  const response = await fetch(`/api/analyze/${encodeURIComponent(name)}`);
  const result = await response.json();
  if (!response.ok) {
    alert(result.error || 'Analysis failed');
    return;
  }
  state.currentData = result;
  state.currentIndex = state.sets.findIndex(item => item.name === name);
  renderSetList();
  renderHeader();
  renderOverview();
  renderModules();
}

async function addSet() {
  const nameInput = document.getElementById('setName');
  const fileInput = document.getElementById('zipFile');
  const setName = nameInput.value.trim();
  const file = fileInput.files[0];
  if (!file) return alert('Choose a zip file');
  const finalSetName = setName || file.name.replace(/\.zip$/i, '');

  const formData = new FormData();
  formData.append('set_name', finalSetName);
  formData.append('zip_file', file);
  setLoading('Uploading and validating the ZIP...');

  const response = await fetch('/api/upload', { method: 'POST', body: formData });
  const result = await response.json();
  if (!response.ok) {
    setLoading('');
    alert(result.error || 'Upload failed');
    return;
  }

  setLoading('Analyzing uploaded set...');
  state.currentData = null;
  clearDashboard();
  await loadSets(result.set_name);
  nameInput.value = '';
  fileInput.value = '';
  setLoading('Upload complete. Result opened successfully.');
}

document.getElementById('addSetBtn').onclick = addSet;
document.getElementById('logoutBtn').onclick = async () => {
  await fetch('/api/logout', { method: 'POST' });
  window.location.href = '/';
};

clearDashboard();
loadSets();
