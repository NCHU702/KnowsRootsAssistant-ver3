const generalTab = document.getElementById('generalTab');
const inheritanceTab = document.getElementById('inheritanceTab');
const generalContent = document.getElementById('generalContent');
const inheritanceContent = document.getElementById('inheritanceContent');
const generalInputContainer = document.getElementById('generalInputContainer');
const inheritanceInputContainer = document.getElementById('inheritanceInputContainer');
const generalInput = document.getElementById('generalInput');
const inheritanceInput = document.getElementById('inheritanceInput');
const generalSubmitBtn = document.getElementById('generalSubmitBtn');
const inheritanceSubmitBtn = document.getElementById('inheritanceSubmitBtn');
const chatTitle = document.getElementById('chatTitle');
const statusIndicator = document.getElementById('statusIndicator');
const errorBox = document.getElementById('errorBox');
const errorMessage = document.getElementById('errorMessage');
const yearsList = document.getElementById('yearsList');
const authorsList = document.getElementById('authorsList');
const researcherSelect = document.getElementById('researcherSelect');

// Load chat histories
let generalHistory = [];
let inheritanceHistory = [];

// Auto-resize textareas
[generalInput, inheritanceInput].forEach(input => {
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
});

// Enable Enter key submission
generalInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitGeneralQuery();
    }
});

inheritanceInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        submitInheritanceQuery();
    }
});

// Check server status and load catalog on page load
checkServerStatus();
loadInheritanceCatalog();

async function checkServerStatus() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        if (data.status === 'ok') {
            statusIndicator.classList.add('status-online');
            statusIndicator.classList.remove('status-offline');
            statusIndicator.querySelector('span').textContent = 'System Online';
        } else {
            statusIndicator.classList.add('status-offline');
            statusIndicator.classList.remove('status-online');
            statusIndicator.querySelector('span').textContent = data.message || 'System Offline';
        }
    } catch (error) {
        statusIndicator.classList.add('status-offline');
        statusIndicator.classList.remove('status-online');
        statusIndicator.querySelector('span').textContent = 'Cannot connect';
    }
}

async function loadInheritanceCatalog() {
    try {
        const response = await fetch('/inheritance_catalog');
        const data = await response.json();
        if (response.ok) {
            // Populate years
            if (yearsList) {
                yearsList.innerHTML = data.available_years.map(year => 
                    `<div class="catalog-item">${year}</div>`
                ).join('');
            }
            
            // Populate authors with "研究生" label
            if (authorsList) {
                authorsList.innerHTML = data.authors.map(author => 
                    `<div class="catalog-item">${author} 研究生</div>`
                ).join('');
            }
            
            // Populate researcher select dropdown
            if (researcherSelect) {
                researcherSelect.innerHTML = '<option value="">Select a researcher...</option>' + 
                    data.authors.map(author => {
                        const name = author.split(' ').slice(1).join(' ');
                        return `<option value="${name}">${author}</option>`;
                    }).join('');
            }
        } else {
            showError(data.error || 'Failed to load catalog');
        }
    } catch (error) {
        showError('Failed to load catalog: ' + error.message);
    }
}

function populateInheritanceInput() {
    inheritanceInput.value = researcherSelect.value;
    inheritanceInput.style.height = 'auto';
    inheritanceInput.style.height = (inheritanceInput.scrollHeight) + 'px';
}

function markdownToHtml(text) {
    text = text.replace(/&/g, '&amp;')
                .replace(/</g, '&lt;')
                .replace(/>/g, '&gt;');
    
    text = text.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    text = text.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    text = text.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    
    text = text.replace(/^---$/gm, '<hr>');
    text = text.replace(/^\*\*\*$/gm, '<hr>');
    
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');
    
    const lines = text.split('\n');
    const result = [];
    let inList = false;
    let listType = null;
    
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        
        const ulMatch = line.match(/^\s*[-*+]\s+(.+)$/);
        if (ulMatch) {
            if (!inList || listType !== 'ul') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ul>');
                inList = true;
                listType = 'ul';
            }
            result.push(`<li>${ulMatch[1]}</li>`);
            continue;
        }
        
        const olMatch = line.match(/^\s*\d+\.\s+(.+)$/);
        if (olMatch) {
            if (!inList || listType !== 'ol') {
                if (inList) result.push(`</${listType}>`);
                result.push('<ol>');
                inList = true;
                listType = 'ol';
            }
            result.push(`<li>${olMatch[1]}</li>`);
            continue;
        }
        
        if (inList) {
            result.push(`</${listType}>`);
            inList = false;
            listType = null;
        }
        
        result.push(line);
    }
    
    if (inList) {
        result.push(`</${listType}>`);
    }
    
    text = result.join('\n');
    text = text.replace(/\n(?!<\/?(h[1-3]|ul|ol|li|hr|p))/g, '<br>');
    text = '<p>' + text + '</p>';
    text = text.replace(/<p><\/p>/g, '');
    text = text.replace(/<p>(<h[1-3]>)/g, '$1');
    text = text.replace(/(<\/h[1-3]>)<\/p>/g, '$1');
    text = text.replace(/<p>(<ul>)/g, '$1');
    text = text.replace(/(<\/ul>)<\/p>/g, '$1');
    text = text.replace(/<p>(<ol>)/g, '$1');
    text = text.replace(/(<\/ol>)<\/p>/g, '$1');
    text = text.replace(/<p>(<hr>)<\/p>/g, '$1');
    
    return text;
}

function addMessageToUI(container, role, content, action = null, save = true, historyArray) {
    const emptyState = container.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (role === 'assistant') {
        contentDiv.innerHTML = markdownToHtml(content);
    } else {
        contentDiv.textContent = content;
    }
    
    messageDiv.appendChild(contentDiv);
    
    if (action && role === 'assistant') {
        const actionDiv = document.createElement('div');
        actionDiv.className = 'message-action';
        actionDiv.textContent = `Action: ${action}`;
        messageDiv.appendChild(actionDiv);
    }
    
    container.appendChild(messageDiv);
    container.scrollTop = container.scrollHeight;

    if (save) {
        historyArray.push({ role, content, action });
    }
}

function addProcessingMessage(container, message = 'Processing...') {
    const processingDiv = document.createElement('div');
    processingDiv.className = 'message message-assistant';
    processingDiv.id = `processingMessage-${container.id}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-processing';
    contentDiv.innerHTML = `
        <div class="spinner"></div>
        <span>${message}</span>
    `;
    
    processingDiv.appendChild(contentDiv);
    container.appendChild(processingDiv);
    container.scrollTop = container.scrollHeight;
}

function updateProcessingMessage(container, message) {
    const processingMsg = container.querySelector(`#processingMessage-${container.id}`);
    if (processingMsg) {
        const span = processingMsg.querySelector('span');
        if (span) span.textContent = message;
    }
}

function removeProcessingMessage(container) {
    const processingMsg = container.querySelector(`#processingMessage-${container.id}`);
    if (processingMsg) processingMsg.remove();
}

function showError(message) {
    errorMessage.textContent = message;
    errorBox.style.display = 'flex';
    setTimeout(() => errorBox.style.display = 'none', 5000);
}

async function submitGeneralQuery() {
    const query = generalInput.value.trim();
    if (!query) return;

    addMessageToUI(generalContent, 'user', query, null, true, generalHistory);
    generalInput.value = '';
    generalInput.style.height = 'auto';
    generalSubmitBtn.disabled = true;

    addProcessingMessage(generalContent, 'Analyzing query...');

    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: query })
        });

        const data = await response.json();

        if (response.ok) {
            if (data.action) {
                updateProcessingMessage(generalContent, `Executing: ${data.action}`);
                await new Promise(resolve => setTimeout(resolve, 800));
            }
            
            removeProcessingMessage(generalContent);
            addMessageToUI(generalContent, 'assistant', data.output, data.action, true, generalHistory);
        } else {
            removeProcessingMessage(generalContent);
            showError(data.error || 'An error occurred');
        }
    } catch (error) {
        removeProcessingMessage(generalContent);
        showError('Failed to connect: ' + error.message);
    } finally {
        generalSubmitBtn.disabled = false;
        generalInput.focus();
    }
}

async function submitInheritanceQuery() {
    const query = inheritanceInput.value.trim();
    if (!query) return;

    addMessageToUI(inheritanceContent, 'user', query, null, true, inheritanceHistory);
    inheritanceInput.value = '';
    inheritanceInput.style.height = 'auto';
    researcherSelect.value = '';
    inheritanceSubmitBtn.disabled = true;

    addProcessingMessage(inheritanceContent, 'Analyzing lineage...');

    try {
        const response = await fetch('/inheritance_query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input: query })
        });

        const data = await response.json();

        if (response.ok) {
            removeProcessingMessage(inheritanceContent);
            addMessageToUI(inheritanceContent, 'assistant', data.output, null, true, inheritanceHistory);
        } else {
            removeProcessingMessage(inheritanceContent);
            showError(data.error || 'An error occurred');
        }
    } catch (error) {
        removeProcessingMessage(inheritanceContent);
        showError('Failed to connect: ' + error.message);
    } finally {
        inheritanceSubmitBtn.disabled = false;
        inheritanceInput.focus();
    }
}

function switchTab(tab) {
    // Hide all contents and inputs
    ['general', 'inheritance', 'categories', 'upload'].forEach(t => {
        const tabEl = document.getElementById(`${t}Tab`);
        const contentEl = document.getElementById(`${t}Content`);
        const inputEl = document.getElementById(`${t}InputContainer`);
        
        if (tabEl) tabEl.classList.remove('active');
        if (contentEl) contentEl.style.display = 'none';
        if (inputEl) inputEl.style.display = 'none';
    });

    // Update sidebar active state
    const sidebarItems = document.querySelectorAll('.function-item');
    sidebarItems.forEach((item, index) => {
        item.classList.remove('active');
        const tabNames = ['general', 'inheritance', 'categories', 'upload'];
        if (tabNames[index] === tab) {
            item.classList.add('active');
        }
    });

    // Show selected tab
    const selectedTab = document.getElementById(`${tab}Tab`);
    const selectedContent = document.getElementById(`${tab}Content`);
    const selectedInput = document.getElementById(`${tab}InputContainer`);
    
    if (selectedTab) selectedTab.classList.add('active');
    if (selectedContent) {
        selectedContent.style.display = 'block';
        // Trigger reflow for animation
        selectedContent.style.animation = 'none';
        selectedContent.offsetHeight; // trigger reflow
        selectedContent.style.animation = null;
    }
    
    // Only show input for general and inheritance
    if ((tab === 'general' || tab === 'inheritance') && selectedInput) {
        selectedInput.style.display = 'block';
        // Trigger animation
        selectedInput.style.animation = 'none';
        selectedInput.offsetHeight;
        selectedInput.style.animation = null;
    }

    // Reload data when switching to specific tabs
    if (tab === 'categories') {
        loadCategoriesData();
    } else if (tab === 'inheritance') {
        loadInheritanceCatalog();
    }

    // Update title with icon
    const titles = {
        general: '<i class="fas fa-comments"></i> General Assistant',
        inheritance: '<i class="fas fa-project-diagram"></i> Inheritance Analysis',
        categories: '<i class="fas fa-search"></i> Data Categories Search',
        upload: '<i class="fas fa-cloud-upload-alt"></i> Add New Paper'
    };
    chatTitle.innerHTML = titles[tab] || '<i class="fas fa-robot"></i> AI Agent Router';
}


function clearChat() {
    if (confirm('Clear chat history for current tab?')) {
        const activeTab = generalTab.classList.contains('active') ? 'general' : 'inheritance';
        if (activeTab === 'general') {
            generalHistory = [];
            generalContent.innerHTML = `
                <div class="empty-state">
                    <div class="empty-state-title">Welcome to General Assistant</div>
                    <div class="empty-state-subtitle">Ask about research papers, web searches, or general queries (e.g., 'Search for remote sensing papers 2024').</div>
                </div>
            `;
        } else {
            inheritanceHistory = [];
            inheritanceContent.innerHTML = `
                <div class="catalog-section">
                    <div class="catalog-title">Available Years</div>
                    <div class="catalog-list" id="yearsList"></div>
                </div>
                <div class="catalog-section">
                    <div class="catalog-title">Available Researchers</div>
                    <div class="catalog-list" id="authorsList"></div>
                </div>
                <div class="empty-state" id="inheritanceEmptyState">
                    <div class="empty-state-title">Research Lineage Analysis</div>
                    <div class="empty-state-subtitle">Select a researcher or enter their name below to analyze their research lineage.</div>
                </div>
            `;
            loadInheritanceCatalog();
        }
    }
}

function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    
    // Save preference to localStorage
    const isDarkMode = document.body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDarkMode);
    
    // Update button icon and text
    const darkModeBtn = document.querySelector('.control-btn[onclick="toggleDarkMode()"]');
    if (isDarkMode) {
        darkModeBtn.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
    } else {
        darkModeBtn.innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
    }
}

// Check for saved dark mode preference on page load
document.addEventListener('DOMContentLoaded', function() {
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    if (savedDarkMode) {
        document.body.classList.add('dark-mode');
        const darkModeBtn = document.querySelector('.control-btn[onclick="toggleDarkMode()"]');
        if (darkModeBtn) {
            darkModeBtn.innerHTML = '<i class="fas fa-sun"></i> Light Mode';
        }
    }
    
    // Set initial active state for sidebar
    const firstFunctionItem = document.querySelector('.function-item');
    if (firstFunctionItem) {
        firstFunctionItem.classList.add('active');
    }
});

// ============= Categories Tab Functions =============
let categoriesData = [];
let currentFilters = {
    purpose: '',
    year: '',
    dataset: '',
    model: ''
};

// Load categories data on page load
loadCategoriesData();

async function loadCategoriesData() {
    try {
        const response = await fetch('/categories_data');
        const data = await response.json();
        if (response.ok) {
            categoriesData = data.papers;
            console.log(`Categories data loaded: ${categoriesData.length} papers`);
            populateInitialPurposes();
            console.log('Initial purposes populated');
        } else {
            showError(data.error || 'Failed to load categories data');
        }
    } catch (error) {
        showError('Failed to load categories data: ' + error.message);
    }
}

function populateInitialPurposes() {
    const purposeFilter = document.getElementById('purposeFilter');
    const purposes = new Set();
    
    categoriesData.forEach(paper => {
        if (paper.研究目的) {
            paper.研究目的.forEach(p => purposes.add(p));
        }
    });
    
    purposeFilter.innerHTML = '<option value="">-- Select Purpose --</option>' +
        Array.from(purposes).sort().map(p => 
            `<option value="${p}">${p}</option>`
        ).join('');
}

function updateYearOptions() {
    const purposeFilter = document.getElementById('purposeFilter');
    const yearFilter = document.getElementById('yearFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    const modelFilter = document.getElementById('modelFilter');
    
    currentFilters.purpose = purposeFilter.value;
    currentFilters.year = '';
    currentFilters.dataset = '';
    currentFilters.model = '';
    
    // Reset dependent dropdowns
    datasetFilter.innerHTML = '<option value="">-- Select Dataset --</option>';
    datasetFilter.disabled = true;
    modelFilter.innerHTML = '<option value="">-- Select Model --</option>';
    modelFilter.disabled = true;
    
    if (!currentFilters.purpose) {
        yearFilter.innerHTML = '<option value="">-- Select Year --</option>';
        yearFilter.disabled = true;
        updateResults();
        return;
    }
    
    // Filter papers by purpose and get available years
    const years = new Set();
    categoriesData.forEach(paper => {
        if (paper.研究目的?.includes(currentFilters.purpose)) {
            if (paper.年份) {
                paper.年份.forEach(y => years.add(y));
            }
        }
    });
    
    yearFilter.innerHTML = '<option value="">-- Select Year --</option>' +
        Array.from(years).sort().reverse().map(y => 
            `<option value="${y}">${y}</option>`
        ).join('');
    yearFilter.disabled = false;
    
    updateResults();
}

function updateDatasetOptions() {
    const datasetFilter = document.getElementById('datasetFilter');
    const modelFilter = document.getElementById('modelFilter');
    const yearFilter = document.getElementById('yearFilter');
    
    currentFilters.year = yearFilter.value;
    currentFilters.dataset = '';
    currentFilters.model = '';
    
    // Reset dependent dropdowns
    modelFilter.innerHTML = '<option value="">-- Select Model --</option>';
    modelFilter.disabled = true;
    
    if (!currentFilters.year) {
        datasetFilter.innerHTML = '<option value="">-- Select Dataset --</option>';
        datasetFilter.disabled = true;
        updateResults();
        return;
    }
    
    // Filter papers by purpose and year, get available datasets
    const datasets = new Set();
    categoriesData.forEach(paper => {
        if (paper.研究目的?.includes(currentFilters.purpose) &&
            paper.年份?.includes(currentFilters.year)) {
            if (paper.資料集) {
                paper.資料集.forEach(d => datasets.add(d));
            }
        }
    });
    
    datasetFilter.innerHTML = '<option value="">-- Select Dataset --</option>' +
        Array.from(datasets).sort().map(d => 
            `<option value="${d}">${d}</option>`
        ).join('');
    datasetFilter.disabled = false;
    
    updateResults();
}

function updateModelOptions() {
    const modelFilter = document.getElementById('modelFilter');
    const datasetFilter = document.getElementById('datasetFilter');
    
    currentFilters.dataset = datasetFilter.value;
    currentFilters.model = '';
    
    if (!currentFilters.dataset) {
        modelFilter.innerHTML = '<option value="">-- Select Model --</option>';
        modelFilter.disabled = true;
        updateResults();
        return;
    }
    
    // Filter papers by all previous selections, get available models
    const models = new Set();
    categoriesData.forEach(paper => {
        if (paper.研究目的?.includes(currentFilters.purpose) &&
            paper.年份?.includes(currentFilters.year) &&
            paper.資料集?.includes(currentFilters.dataset)) {
            if (paper.建模) {
                paper.建模.forEach(m => models.add(m));
            }
        }
    });
    
    modelFilter.innerHTML = '<option value="">-- Select Model --</option>' +
        Array.from(models).sort().map(m => 
            `<option value="${m}">${m}</option>`
        ).join('');
    modelFilter.disabled = false;
    
    updateResults();
}

function updateResults() {
    const modelFilter = document.getElementById('modelFilter');
    currentFilters.model = modelFilter.value;
    
    // Filter papers based on all current selections
    const filtered = categoriesData.filter(paper => {
        const matchPurpose = !currentFilters.purpose || paper.研究目的?.includes(currentFilters.purpose);
        const matchYear = !currentFilters.year || paper.年份?.includes(currentFilters.year);
        const matchDataset = !currentFilters.dataset || paper.資料集?.includes(currentFilters.dataset);
        const matchModel = !currentFilters.model || paper.建模?.includes(currentFilters.model);
        
        return matchPurpose && matchYear && matchDataset && matchModel;
    });
    
    displayCategoryResults(filtered);
}

function displayCategoryResults(papers) {
    const resultsContainer = document.getElementById('categoriesResults');
    
    if (papers.length === 0) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <div class="empty-state-title">No papers found</div>
                <div class="empty-state-subtitle">Try adjusting your filters</div>
            </div>
        `;
        return;
    }

    const filterSummary = [];
    if (currentFilters.purpose) filterSummary.push(`Purpose: ${currentFilters.purpose}`);
    if (currentFilters.year) filterSummary.push(`Year: ${currentFilters.year}`);
    if (currentFilters.dataset) filterSummary.push(`Dataset: ${currentFilters.dataset}`);
    if (currentFilters.model) filterSummary.push(`Model: ${currentFilters.model}`);

    resultsContainer.innerHTML = `
        <div class="results-header">
            Found ${papers.length} paper${papers.length > 1 ? 's' : ''}
            ${filterSummary.length > 0 ? ' - ' + filterSummary.join(' | ') : ''}
        </div>
    ` + papers.map(paper => `
        <div class="paper-card">
            <div class="paper-title">${paper.論文標題}</div>
            <div class="paper-details">
                ${paper.研究目的?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">研究目的</div>
                        <div class="detail-value">
                            ${paper.研究目的.map(p => `<span class="tag">${p}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${paper.年份?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">年份</div>
                        <div class="detail-value">${paper.年份.join(', ')}</div>
                    </div>
                ` : ''}
                ${paper.建模?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">建模</div>
                        <div class="detail-value">
                            ${paper.建模.map(m => `<span class="tag">${m}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${paper.資料前處理?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">資料前處理</div>
                        <div class="detail-value">
                            ${paper.資料前處理.map(p => `<span class="tag">${p}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${paper.評估指標?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">評估指標</div>
                        <div class="detail-value">
                            ${paper.評估指標.map(i => `<span class="tag">${i}</span>`).join('')}
                        </div>
                    </div>
                ` : ''}
                ${paper.資料集?.length ? `
                    <div class="detail-item">
                        <div class="detail-label">資料集 (${paper.資料集.length})</div>
                        <div class="detail-value">
                            ${paper.資料集.slice(0, 4).map(d => `<span class="tag">${d}</span>`).join('')}
                            ${paper.資料集.length > 4 ? `<span class="tag">+${paper.資料集.length - 4} more</span>` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        </div>
    `).join('');
}

function resetCategoryFilters() {
    currentFilters = {
        purpose: '',
        year: '',
        dataset: '',
        model: ''
    };
    
    document.getElementById('purposeFilter').value = '';
    document.getElementById('yearFilter').innerHTML = '<option value="">-- Select Year --</option>';
    document.getElementById('yearFilter').disabled = true;
    document.getElementById('datasetFilter').innerHTML = '<option value="">-- Select Dataset --</option>';
    document.getElementById('datasetFilter').disabled = true;
    document.getElementById('modelFilter').innerHTML = '<option value="">-- Select Model --</option>';
    document.getElementById('modelFilter').disabled = true;
    document.getElementById('categoriesResults').innerHTML = '';
}

// ============= Upload Tab Functions =============

let uploadedFile = null;
let extractedData = null;

// Initialize upload functionality
const uploadZone = document.getElementById('uploadZone');
const pdfFileInput = document.getElementById('pdfFileInput');
const uploadProgress = document.getElementById('uploadProgress');
const uploadResult = document.getElementById('uploadResult');
const progressText = document.getElementById('progressText');
const metadataPreview = document.getElementById('metadataPreview');

// Click to upload
uploadZone.addEventListener('click', () => {
    pdfFileInput.click();
});

// File selection
pdfFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

// Drag and drop
uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    // Validate file type
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showError('Invalid file type. Only PDF files are accepted.');
        return;
    }
    
    // Validate file size (50MB)
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showError(`File size (${(file.size / 1024 / 1024).toFixed(1)}MB) exceeds maximum limit of 50MB`);
        return;
    }
    
    uploadedFile = file;
    uploadPaper(file);
}

async function uploadPaper(file) {
    // Show progress
    uploadZone.style.display = 'none';
    uploadResult.style.display = 'none';
    uploadProgress.style.display = 'block';
    progressText.textContent = '正在上傳檔案... / Uploading file...';
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Simulate progress updates
    const progressSteps = [
        { delay: 500, text: '正在驗證 PDF 格式... / Validating PDF format...' },
        { delay: 1500, text: '正在提取文字內容... / Extracting text content...' },
        { delay: 3000, text: '正在分析論文結構... / Analyzing paper structure...' },
        { delay: 5000, text: '正在使用 LLM 提取資訊... / Extracting information with LLM...' },
        { delay: 8000, text: '正在分類研究領域... / Classifying research area...' },
        { delay: 11000, text: '正在更新資料庫... / Updating database...' }
    ];
    
    let currentStep = 0;
    const progressInterval = setInterval(() => {
        if (currentStep < progressSteps.length) {
            progressText.textContent = progressSteps[currentStep].text;
            currentStep++;
        }
    }, 2000);
    
    try {
        const response = await fetch('/upload_paper', {
            method: 'POST',
            body: formData
        });
        
        clearInterval(progressInterval);
        progressText.textContent = '正在完成處理... / Finalizing...';
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            // Success - show extracted metadata
            extractedData = data;
            displayMetadata(data);
        } else {
            // Error
            showError(data.error || 'Upload failed');
            resetUploadForm();
        }
    } catch (error) {
        clearInterval(progressInterval);
        showError('Failed to connect to server: ' + error.message);
        resetUploadForm();
    }
}

function displayMetadata(data) {
    uploadProgress.style.display = 'none';
    uploadResult.style.display = 'block';
    
    const paperData = data.paper_data;
    const classification = data.classification;
    
    let html = '<form id="metadataForm" style="width: 100%;">';
    
    // Paper Title (required, single line input)
    html += `<div class="metadata-item">
        <div class="metadata-label">論文標題 / Title: <span style="color: red;">*</span></div>
        <input type="text" id="edit_論文標題" class="metadata-input" 
               value="${paperData.論文標題 || ''}" required 
               style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px;">
    </div>`;
    
    // Year (comma-separated)
    html += `<div class="metadata-item">
        <div class="metadata-label">年份 / Year: <small style="color: #718096;">(comma separated)</small></div>
        <input type="text" id="edit_年份" class="metadata-input" 
               value="${paperData.年份?.join(', ') || ''}" 
               placeholder="e.g., 2024"
               style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px;">
    </div>`;
    
    // Authors (comma-separated)
    html += `<div class="metadata-item">
        <div class="metadata-label">作者 / Authors: <small style="color: #718096;">(comma separated)</small></div>
        <input type="text" id="edit_作者" class="metadata-input" 
               value="${paperData.作者?.join(', ') || ''}" 
               placeholder="e.g., John Doe, Jane Smith"
               style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px;">
    </div>`;
    
    // Research Purpose (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">研究目的 / Research Purpose: <small style="color: #718096;">(one per line)</small></div>
        <textarea id="edit_研究目的" class="metadata-input" rows="3" 
                  placeholder="Enter each purpose on a new line"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.研究目的?.join('\n') || ''}</textarea>
    </div>`;
    
    // Datasets (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">資料集 / Datasets: <small style="color: #718096;">(one per line)</small></div>
        <textarea id="edit_資料集" class="metadata-input" rows="2" 
                  placeholder="Enter each dataset on a new line"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.資料集?.join('\n') || ''}</textarea>
    </div>`;
    
    // Preprocessing (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">資料前處理 / Preprocessing: <small style="color: #718096;">(one per line)</small></div>
        <textarea id="edit_資料前處理" class="metadata-input" rows="2" 
                  placeholder="Enter each method on a new line"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.資料前處理?.join('\n') || ''}</textarea>
    </div>`;
    
    // Modeling (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">建模 / Modeling: <small style="color: #718096;">(one per line)</small></div>
        <textarea id="edit_建模" class="metadata-input" rows="2" 
                  placeholder="Enter each model on a new line"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.建模?.join('\n') || ''}</textarea>
    </div>`;
    
    // Metrics (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">評估指標 / Metrics: <small style="color: #718096;">(one per line)</small></div>
        <textarea id="edit_評估指標" class="metadata-input" rows="2" 
                  placeholder="Enter each metric on a new line"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.評估指標?.join('\n') || ''}</textarea>
    </div>`;
    
    // Other (textarea, one per line)
    html += `<div class="metadata-item">
        <div class="metadata-label">其他 / Other: <small style="color: #718096;">(one per line, optional)</small></div>
        <textarea id="edit_其他" class="metadata-input" rows="2" 
                  placeholder="Enter any other information"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; font-family: inherit;">${paperData.其他?.join('\n') || ''}</textarea>
    </div>`;
    
    // Classification (textarea, one per line: area, trunk, reasoning)
    const classificationText = classification ? `${classification.area_name || ''}\n${classification.trunk_name || ''}\n${classification.reasoning || ''}` : '';
    html += `<div class="metadata-item">
        <div class="metadata-label">研究分類 / Classification: <small style="color: #718096;">(one per line: area, trunk, reasoning)</small></div>
        <textarea id="edit_classification" class="metadata-input" rows="3"
                  placeholder="Line 1: Area name\nLine 2: Trunk name\nLine 3: Reasoning"
                  style="width: 100%; padding: 8px; border: 1px solid #cbd5e0; border-radius: 4px; background: #ffffff; color: #2f2d2a !important; font-family: inherit;">${classificationText}</textarea>
    </div>`;
    
    // Stored filename (hidden)
    html += `<input type="hidden" id="stored_filename" value="${data.stored_filename}">`;
    
    html += '</form>';
    
    metadataPreview.innerHTML = html;
}

async function confirmUpload() {
    // Collect edited data from form
    const editedData = {
        論文標題: document.getElementById('edit_論文標題').value.trim(),
        年份: document.getElementById('edit_年份').value.split(',').map(s => s.trim()).filter(s => s),
        作者: document.getElementById('edit_作者').value.split(',').map(s => s.trim()).filter(s => s),
        研究目的: document.getElementById('edit_研究目的').value.split('\n').map(s => s.trim()).filter(s => s),
        資料集: document.getElementById('edit_資料集').value.split('\n').map(s => s.trim()).filter(s => s),
        資料前處理: document.getElementById('edit_資料前處理').value.split('\n').map(s => s.trim()).filter(s => s),
        建模: document.getElementById('edit_建模').value.split('\n').map(s => s.trim()).filter(s => s),
        評估指標: document.getElementById('edit_評估指標').value.split('\n').map(s => s.trim()).filter(s => s),
        其他: document.getElementById('edit_其他').value.split('\n').map(s => s.trim()).filter(s => s)
    };

    // Collect classification data
    const classificationLines = document.getElementById('edit_classification') ?
        document.getElementById('edit_classification').value.split('\n').map(s => s.trim()) : [];
    const editedClassification = {
        area_name: classificationLines[0] || '',
        trunk_name: classificationLines[1] || '',
        reasoning: classificationLines.slice(2).join('\n') || ''
    };
    
    // Validate title
    if (!editedData.論文標題) {
        alert('論文標題為必填項目！\nPaper title is required!');
        return;
    }
    
    // Check if data or classification was modified
    const originalData = extractedData.paper_data;
    const originalClassification = extractedData.classification || {};
    const isModifiedData = JSON.stringify(editedData) !== JSON.stringify(originalData);
    const isModifiedClassification = JSON.stringify(editedClassification) !== JSON.stringify(originalClassification);
    const isModified = isModifiedData || isModifiedClassification;
    
    if (isModified) {
        // Show saving progress
        uploadProgress.style.display = 'block';
        uploadResult.style.display = 'none';
        progressText.textContent = '正在更新資料... / Updating data...';
        
        try {
            const response = await fetch('/update_paper', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    paper_data: editedData,
                    original_title: originalData.論文標題,
                    stored_filename: document.getElementById('stored_filename').value,
                    classification: editedClassification
                })
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                alert('論文資料已成功更新！\nPaper data updated successfully!');
                resetUploadForm();
                // Always refresh catalog and categories data
                try {
                    loadInheritanceCatalog();
                    loadCategoriesData();
                } catch (e) {
                    console.log('Data refresh failed:', e.message);
                }
            } else {
                throw new Error(data.error || 'Update failed');
            }
        } catch (error) {
            alert('更新失敗：' + error.message + '\nUpdate failed: ' + error.message);
            uploadProgress.style.display = 'none';
            uploadResult.style.display = 'block';
        }
    } else {
        // No changes, just confirm
        alert('論文已成功加入資料庫！\nPaper successfully added to database!');
        resetUploadForm();
        // Always refresh catalog and categories data
        try {
            loadInheritanceCatalog();
            loadCategoriesData();
        } catch (e) {
            console.log('Data refresh failed:', e.message);
        }
    }
}

async function cancelUpload() {
    if (!confirm('確定要取消嗎？已添加的論文資料將被刪除。\nAre you sure you want to cancel? The paper data will be removed from the database.')) {
        return;
    }
    
    // Show progress
    uploadProgress.style.display = 'block';
    uploadResult.style.display = 'none';
    progressText.textContent = '正在刪除資料... / Removing data...';
    
    try {
        const response = await fetch('/cancel_upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                paper_title: extractedData.paper_data.論文標題,
                stored_filename: extractedData.stored_filename
            })
        });
        
        const data = await response.json();
        
        if (response.ok && data.success) {
            alert('取消成功！論文資料已從資料庫中刪除。\nCancellation successful! Paper data has been removed from the database.');
        } else {
            alert('警告：取消時發生錯誤，部分資料可能未刪除。\nWarning: Error occurred during cancellation, some data may not be removed.');
        }
    } catch (error) {
        alert('錯誤：無法連接伺服器刪除資料。\nError: Failed to connect to server for data removal.');
    } finally {
        resetUploadForm();
    }
}

function resetUploadForm() {
    uploadedFile = null;
    extractedData = null;
    pdfFileInput.value = '';
    
    uploadProgress.style.display = 'none';
    uploadResult.style.display = 'none';
    uploadZone.style.display = 'block';
    metadataPreview.innerHTML = '';
}
