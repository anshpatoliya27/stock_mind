/**
 * StockMind AI — Front-End Application Logic
 * Institutional equity intelligence terminal with multimodal chart recognition,
 * live market ticker tape, Grok/Gemini/OpenAI API configuration, and big data grounding.
 */

// Global Application State
const state = {
  currentConversationId: null,
  attachedImageFile: null,
  isStreaming: false,
  systemStatus: null,
  tickerTimer: null,
};

// DOM Element Selectors
const DOM = {
  // Sidebar & Navigation
  sidebar: document.getElementById('sidebar'),
  menuToggle: document.getElementById('menuToggle'),
  mobileMenuBtn: document.getElementById('mobileMenuBtn'),
  newChatBtn: document.getElementById('newChatBtn'),
  conversationsList: document.getElementById('conversationsList'),
  watchlistContainer: document.getElementById('watchlistContainer'),
  sidebarApiKeyBtn: document.getElementById('sidebarApiKeyBtn'),
  configStatusBtn: document.getElementById('configStatusBtn'),
  statusDot: document.getElementById('statusDot'),
  statusLabel: document.getElementById('statusLabel'),

  // Header & Ticker Tape
  tickerItemsContainer: document.getElementById('tickerItemsContainer'),
  currentChatTitle: document.getElementById('currentChatTitle'),
  topConnectApiKeyBtn: document.getElementById('topConnectApiKeyBtn'),
  topApiKeyBtnText: document.getElementById('topApiKeyBtnText'),
  refreshBtn: document.getElementById('refreshBtn'),

  // Center Content Flow
  mainContentFlow: document.getElementById('mainContentFlow'),
  welcomeContainer: document.getElementById('welcomeContainer'),
  chatMessages: document.getElementById('chatMessages'),
  heroDropzone: document.getElementById('heroDropzone'),

  // Floating Input Dock
  chatForm: document.getElementById('chatForm'),
  messageInput: document.getElementById('messageInput'),
  sendBtn: document.getElementById('sendBtn'),
  attachBtn: document.getElementById('attachBtn'),
  imageFileInput: document.getElementById('imageFileInput'),
  imagePreviewBar: document.getElementById('imagePreviewBar'),
  imagePreviewImg: document.getElementById('imagePreviewImg'),
  previewFilename: document.getElementById('previewFilename'),
  removeImgBtn: document.getElementById('removeImgBtn'),
  thinkBtn: document.getElementById('thinkBtn'),

  // API Key Settings Modal
  apiKeyModalBackdrop: document.getElementById('apiKeyModalBackdrop'),
  apiKeyModalCloseBtn: document.getElementById('apiKeyModalCloseBtn'),
  cancelApiKeyBtn: document.getElementById('cancelApiKeyBtn'),
  apiSettingsForm: document.getElementById('apiSettingsForm'),
  providerSelect: document.getElementById('providerSelect'),
  apiKeyInput: document.getElementById('apiKeyInput'),
  apiKeyLabelText: document.getElementById('apiKeyLabelText'),
  keyPortalLink: document.getElementById('keyPortalLink'),
  toggleKeyVisibilityBtn: document.getElementById('toggleKeyVisibilityBtn'),
  keyFieldHint: document.getElementById('keyFieldHint'),
  modelSelect: document.getElementById('modelSelect'),
  saveApiKeyBtn: document.getElementById('saveApiKeyBtn'),
  saveBtnText: document.getElementById('saveBtnText'),

  // System Diagnostics Modal
  modalBackdrop: document.getElementById('modalBackdrop'),
  modalCloseBtn: document.getElementById('modalCloseBtn'),
  modalBody: document.getElementById('modalBody'),
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
  setupEventListeners();
  setupDragAndDrop();
  await loadMarketTickerTape();
  await checkSystemStatus();
  await loadConversations();

  // Periodically refresh market ticker tape every 60s
  state.tickerTimer = setInterval(loadMarketTickerTape, 60000);
});

// ============================================================================
// Event Listeners Setup
// ============================================================================

function setupEventListeners() {
  // Sidebar toggles
  if (DOM.menuToggle) {
    DOM.menuToggle.addEventListener('click', () => {
      DOM.sidebar.classList.toggle('open');
    });
  }

  if (DOM.mobileMenuBtn) {
    DOM.mobileMenuBtn.addEventListener('click', () => {
      DOM.sidebar.classList.toggle('open');
    });
  }

  // New analysis session
  if (DOM.newChatBtn) {
    DOM.newChatBtn.addEventListener('click', () => startNewConversation());
  }

  if (DOM.refreshBtn) {
    DOM.refreshBtn.addEventListener('click', () => startNewConversation());
  }

  // Watchlist clicks
  if (DOM.watchlistContainer) {
    DOM.watchlistContainer.addEventListener('click', (e) => {
      const chip = e.target.closest('.watchlist-chip, .wl-chip');
      if (chip) {
        const ticker = chip.getAttribute('data-ticker');
        triggerAnalysisQuery(`Generate institutional technical analysis, floor pivots, and future price targets for ${ticker}.`);
      }
    });
  }

  // Welcome Action Cards (ChatGPT style)
  document.querySelectorAll('.action-card, .suggestion-card').forEach((card) => {
    card.addEventListener('click', () => {
      const prompt = card.getAttribute('data-prompt');
      if (prompt) {
        DOM.messageInput.value = prompt;
        autoResizeTextarea(DOM.messageInput);
        DOM.messageInput.focus();
      }
    });
  });

  // Quant Reasoning Think Toggle
  if (DOM.thinkBtn) {
    DOM.thinkBtn.addEventListener('click', () => {
      const current = DOM.messageInput.value;
      if (!current.includes('Please perform detailed reasoning')) {
        DOM.messageInput.value = (
          current + ' Please perform detailed quantitative reasoning on technical indicators, moving averages, and support/resistance levels.'
        ).trim();
      }
      autoResizeTextarea(DOM.messageInput);
      DOM.messageInput.focus();
    });
  }

  // Image upload triggers
  if (DOM.attachBtn && DOM.imageFileInput) {
    DOM.attachBtn.addEventListener('click', () => DOM.imageFileInput.click());
  }

  if (DOM.imageFileInput) {
    DOM.imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleImageSelected(e.target.files[0]);
      }
    });
  }

  if (DOM.removeImgBtn) {
    DOM.removeImgBtn.addEventListener('click', () => clearAttachedImage());
  }

  // Auto-resize textarea
  if (DOM.messageInput) {
    DOM.messageInput.addEventListener('input', () => autoResizeTextarea(DOM.messageInput));
    DOM.messageInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleFormSubmit();
      }
    });
  }

  // Form submit
  if (DOM.chatForm) {
    DOM.chatForm.addEventListener('submit', (e) => {
      e.preventDefault();
      handleFormSubmit();
    });
  }

  // API Key Settings Modal
  if (DOM.topConnectApiKeyBtn) {
    DOM.topConnectApiKeyBtn.addEventListener('click', () => openApiKeyModal());
  }

  if (DOM.sidebarApiKeyBtn) {
    DOM.sidebarApiKeyBtn.addEventListener('click', () => openApiKeyModal());
  }

  if (DOM.apiKeyModalCloseBtn) {
    DOM.apiKeyModalCloseBtn.addEventListener('click', () => closeApiKeyModal());
  }

  if (DOM.cancelApiKeyBtn) {
    DOM.cancelApiKeyBtn.addEventListener('click', () => closeApiKeyModal());
  }

  if (DOM.providerSelect) {
    DOM.providerSelect.addEventListener('change', () => handleProviderSelectChange());
  }

  if (DOM.toggleKeyVisibilityBtn) {
    DOM.toggleKeyVisibilityBtn.addEventListener('click', () => {
      const isPwd = DOM.apiKeyInput.type === 'password';
      DOM.apiKeyInput.type = isPwd ? 'text' : 'password';
      DOM.toggleKeyVisibilityBtn.textContent = isPwd ? '🔒' : '👁️';
    });
  }

  if (DOM.apiSettingsForm) {
    DOM.apiSettingsForm.addEventListener('submit', handleSaveApiSettings);
  }

  // System Diagnostics Modal
  if (DOM.configStatusBtn) {
    DOM.configStatusBtn.addEventListener('click', () => openDiagnosticsModal());
  }

  if (DOM.modalCloseBtn) {
    DOM.modalCloseBtn.addEventListener('click', () => closeDiagnosticsModal());
  }

  // Close modals on clicking backdrop
  window.addEventListener('click', (e) => {
    if (e.target === DOM.apiKeyModalBackdrop) closeApiKeyModal();
    if (e.target === DOM.modalBackdrop) closeDiagnosticsModal();
  });
}

function autoResizeTextarea(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
}

// ============================================================================
// Live Market Ticker Tape Fetch & Render
// ============================================================================

async function loadMarketTickerTape() {
  try {
    const res = await fetch('/api/market/ticker_tape');
    if (!res.ok) return;
    const data = await res.json();
    const items = Array.isArray(data) ? data : (data.items || []);
    if (!items.length || !DOM.tickerItemsContainer) return;

    DOM.tickerItemsContainer.innerHTML = '';
    items.forEach((item) => {
      const span = document.createElement('span');
      span.className = 'ticker-item';
      const isUp = item.positive !== undefined ? item.positive : String(item.change).startsWith('+');
      const isDown = !isUp && String(item.change).startsWith('-');
      const chgClass = isUp ? 'up' : (isDown ? 'down' : 'neutral');
      const priceText = item.price || item.price_display || '—';
      const chgText = item.change || item.change_display || '0.00%';

      span.innerHTML = `
        <span class="ticker-sym">${item.symbol}</span>
        <span class="ticker-val">${priceText}</span>
        <span class="ticker-chg ${chgClass}">${chgText}</span>
      `;
      span.style.cursor = 'pointer';
      span.title = `Click to analyze ${item.symbol}`;
      span.addEventListener('click', () => {
        triggerAnalysisQuery(`Generate comprehensive technical chart analysis and multi-horizon target levels for ${item.symbol}.`);
      });
      DOM.tickerItemsContainer.appendChild(span);
    });
  } catch (err) {
    console.warn('Ticker tape fetch warning:', err);
  }
}

function triggerAnalysisQuery(query) {
  DOM.messageInput.value = query;
  autoResizeTextarea(DOM.messageInput);
  handleFormSubmit();
}

// ============================================================================
// Drag and Drop Chart Upload
// ============================================================================

function setupDragAndDrop() {
  const dropzone = DOM.heroDropzone;
  if (dropzone) {
    dropzone.addEventListener('click', () => {
      if (DOM.imageFileInput) DOM.imageFileInput.click();
    });

    ['dragenter', 'dragover'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleImageSelected(e.dataTransfer.files[0]);
      }
    });
  }

  // Global window drop prevention
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageSelected(e.dataTransfer.files[0]);
    }
  });
}

function handleImageSelected(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a valid trading chart image (PNG, JPG, or WEBP).');
    return;
  }
  state.attachedImageFile = file;
  DOM.previewFilename.textContent = file.name;

  const reader = new FileReader();
  reader.onload = (e) => {
    DOM.imagePreviewImg.src = e.target.result;
    DOM.imagePreviewBar.style.display = 'flex';
  };
  reader.readAsDataURL(file);
}

function clearAttachedImage() {
  state.attachedImageFile = null;
  DOM.imageFileInput.value = '';
  DOM.imagePreviewBar.style.display = 'none';
  DOM.imagePreviewImg.src = '';
}

// ============================================================================
// API Key & Model Configuration Modal
// ============================================================================

function openApiKeyModal() {
  DOM.apiKeyModalBackdrop.style.display = 'flex';
  const llm = (state.systemStatus && state.systemStatus.llm) || {};
  const currentProvider = (llm.provider || 'gemini').toLowerCase();

  DOM.providerSelect.value = currentProvider;
  handleProviderSelectChange();

  if (llm.key_preview && llm.key_preview !== 'None') {
    DOM.apiKeyInput.placeholder = `Current: ${llm.key_preview}`;
  } else {
    DOM.apiKeyInput.placeholder = currentProvider === 'gemini' ? 'AIzaSy... or API key' : 'Key...';
  }
  DOM.apiKeyInput.focus();
}

function closeApiKeyModal() {
  DOM.apiKeyModalBackdrop.style.display = 'none';
  DOM.apiKeyInput.value = '';
}

function handleProviderSelectChange() {
  const provider = DOM.providerSelect.value;
  DOM.modelSelect.innerHTML = '';

  if (provider === 'gemini') {
    DOM.apiKeyLabelText.textContent = 'Google Gemini API Key';
    DOM.keyPortalLink.href = 'https://aistudio.google.com/app/apikey';
    DOM.keyPortalLink.textContent = 'Get Gemini Key from Google AI Studio →';
    DOM.keyFieldHint.innerHTML = 'Paste your Google Gemini API key. Enables real-time multimodal reasoning and dynamic answers.';

    addModelOption('gemini-3.5-flash', 'gemini-3.5-flash (Fast, Vision & Multimodal Reasoning)', true);
    addModelOption('gemini-flash-lite-latest', 'gemini-flash-lite-latest (Ultra Low Latency)', false);
  } else if (provider === 'grok') {
    DOM.apiKeyLabelText.textContent = 'xAI Grok API Key';
    DOM.keyPortalLink.href = 'https://console.x.ai/';
    DOM.keyPortalLink.textContent = 'Get xAI Key from console.x.ai →';
    DOM.keyFieldHint.innerHTML = 'Paste your <code>xai-...</code> API key. xAI provides high-precision multimodal reasoning.';
    
    addModelOption('grok-2-vision-1212', 'grok-2-vision-1212 (Multimodal Vision & Stock Reasoning)', true);
    addModelOption('grok-2', 'grok-2 (High Precision Quantitative Text)', false);
  } else {
    DOM.apiKeyLabelText.textContent = 'OpenAI API Key';
    DOM.keyPortalLink.href = 'https://platform.openai.com/api-keys';
    DOM.keyPortalLink.textContent = 'Get OpenAI Key →';
    DOM.keyFieldHint.innerHTML = 'Paste your OpenAI <code>sk-...</code> API key.';

    addModelOption('gpt-4o-mini', 'gpt-4o-mini (Vision & Reasoning)', true);
    addModelOption('gpt-4o', 'gpt-4o (Flagship Multimodal)', false);
  }
}

function addModelOption(val, text, selected = false) {
  const opt = document.createElement('option');
  opt.value = val;
  opt.textContent = text;
  if (selected) opt.selected = true;
  DOM.modelSelect.appendChild(opt);
}

async function handleSaveApiSettings(e) {
  e.preventDefault();
  const provider = DOM.providerSelect.value;
  const apiKey = DOM.apiKeyInput.value.trim();
  const model = DOM.modelSelect.value;

  DOM.saveBtnText.textContent = 'Saving & Connecting...';
  DOM.saveApiKeyBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('provider', provider);
    formData.append('api_key', apiKey);
    formData.append('model', model);

    const res = await fetch('/api/settings', {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    await checkSystemStatus();
    closeApiKeyModal();
    alert(`Connected successfully to ${provider.toUpperCase()} (${data.model})!`);
  } catch (err) {
    console.error('Settings save failed:', err);
    alert(`Failed to save settings: ${err.message}`);
  } finally {
    DOM.saveBtnText.textContent = 'Save & Connect';
    DOM.saveApiKeyBtn.disabled = false;
  }
}

// ============================================================================
// System Status & Diagnostics Modal
// ============================================================================

async function checkSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status failed');
    const data = await res.json();
    state.systemStatus = data;

    const llm = data.llm || {};
    if (llm.is_configured) {
      DOM.statusDot.className = 'status-dot';
      DOM.statusLabel.textContent = `${llm.provider.toUpperCase()} Active`;
      DOM.topApiKeyBtnText.textContent = `${llm.provider.toUpperCase()} Connected`;
      DOM.topConnectApiKeyBtn.classList.add('connected');
    } else {
      DOM.statusDot.className = 'status-dot warning';
      DOM.statusLabel.textContent = 'Local Engine (Gemini Ready)';
      DOM.topApiKeyBtnText.textContent = 'Connect Gemini Key';
      DOM.topConnectApiKeyBtn.classList.remove('connected');
    }
  } catch (err) {
    console.warn('System status check error:', err);
    DOM.statusDot.className = 'status-dot warning';
    DOM.statusLabel.textContent = 'Local Engine Active';
  }
}

function openDiagnosticsModal() {
  DOM.modalBackdrop.style.display = 'flex';
  const s = state.systemStatus || {};
  const llm = s.llm || {};
  const ds = s.dataset || {};
  const stats = ds.stats || {};

  DOM.modalBody.innerHTML = `
    <div class="diag-row">
      <span class="diag-lbl">AI Engine:</span>
      <span class="diag-val">${(llm.provider || 'Google Gemini').toUpperCase()}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Model:</span>
      <span class="diag-val">${llm.model || 'gemini-3.5-flash'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Cloud API Key:</span>
      <span class="diag-val">
        ${llm.is_configured ? 'Connected & Active' : 'Not configured (Local mode)'}
      </span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Key Preview:</span>
      <span class="diag-val">${llm.key_preview || 'None'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">NSE Daily Trades:</span>
      <span class="diag-val">${stats.daily_trades || '3,919,007 rows (2019–2026)'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Companies Tracked:</span>
      <span class="diag-val">${stats.companies || '5,028 ticker archives'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Minute-Level Ticks:</span>
      <span class="diag-val">${stats.minute_ticks || '515 files'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Chart Patterns:</span>
      <span class="diag-val">${stats.chart_patterns || '340 patterns'}</span>
    </div>
    <div class="diag-row">
      <span class="diag-lbl">Financial News:</span>
      <span class="diag-val">${stats.financial_news || '26,000 articles'}</span>
    </div>
    ${
      !llm.is_configured
        ? `<div class="offline-note" style="margin-top: 12px;">
            Click <em>Connect Gemini Key</em> at the top to paste your <code>GEMINI_API_KEY</code> and unlock real-time multimodal GenAI.
          </div>`
        : ''
    }
  `;
}

function closeDiagnosticsModal() {
  DOM.modalBackdrop.style.display = 'none';
}

// ============================================================================
// Conversation Management
// ============================================================================

async function loadConversations() {
  try {
    const res = await fetch('/api/conversations');
    if (!res.ok) return;
    const conversations = await res.json();
    renderConversationsList(conversations);
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
}

function renderConversationsList(conversations) {
  DOM.conversationsList.innerHTML = '';
  if (!conversations || conversations.length === 0) {
    const emptySpan = document.createElement('span');
    emptySpan.style.fontSize = '0.75rem';
    emptySpan.style.color = 'var(--text-muted)';
    emptySpan.style.padding = '8px 10px';
    emptySpan.textContent = 'No previous analyses';
    DOM.conversationsList.appendChild(emptySpan);
    return;
  }

  conversations.forEach((c) => {
    const item = document.createElement('div');
    item.className = `conv-item ${state.currentConversationId === c.id ? 'active' : ''}`;
    item.setAttribute('data-id', c.id);

    const titleSpan = document.createElement('span');
    titleSpan.className = 'conv-title';
    titleSpan.textContent = c.title || 'Stock Analysis';

    const delBtn = document.createElement('button');
    delBtn.className = 'conv-delete-btn';
    delBtn.title = 'Delete analysis';
    delBtn.innerHTML = `
      <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      </svg>
    `;

    item.appendChild(titleSpan);
    item.appendChild(delBtn);

    item.addEventListener('click', (e) => {
      if (e.target.closest('.conv-delete-btn')) {
        e.stopPropagation();
        deleteConversation(c.id);
      } else {
        selectConversation(c.id);
      }
    });

    DOM.conversationsList.appendChild(item);
  });
}

async function selectConversation(convId) {
  state.currentConversationId = convId;
  DOM.sidebar.classList.remove('open');

  try {
    const res = await fetch(`/api/conversations/${convId}`);
    if (!res.ok) throw new Error('Not found');
    const conv = await res.json();

    DOM.currentChatTitle.textContent = conv.title || 'StockMind Analysis';
    renderMessages(conv.messages || []);
    document.querySelectorAll('.conv-item').forEach((el) => el.classList.remove('active'));
    await loadConversations();
  } catch (err) {
    console.error('Failed to fetch conversation:', err);
  }
}

function startNewConversation() {
  state.currentConversationId = null;
  DOM.currentChatTitle.textContent = 'StockMind AI Intelligence';
  DOM.chatMessages.innerHTML = '';
  DOM.welcomeContainer.style.display = 'flex';
  clearAttachedImage();
  DOM.messageInput.value = '';
  DOM.messageInput.focus();
  document.querySelectorAll('.conv-item').forEach((el) => el.classList.remove('active'));
}

async function deleteConversation(convId) {
  if (!confirm('Are you sure you want to delete this analysis session?')) return;
  try {
    await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
    if (state.currentConversationId === convId) {
      startNewConversation();
    }
    await loadConversations();
  } catch (err) {
    console.error('Delete conversation error:', err);
  }
}

// ============================================================================
// Message Rendering & Streaming
// ============================================================================

function renderMessages(messages) {
  DOM.chatMessages.innerHTML = '';
  if (!messages || messages.length === 0) {
    DOM.welcomeContainer.style.display = 'flex';
    return;
  }

  DOM.welcomeContainer.style.display = 'none';
  messages.forEach((m) => {
    appendMessageElement(m.role, m.content, m.image_url);
  });
  scrollToBottom();
}

function appendMessageElement(role, content, imageUrl = null) {
  DOM.welcomeContainer.style.display = 'none';

  const row = document.createElement('div');
  row.className = `message-row ${role === 'user' ? 'user-row' : 'assistant-row'}`;

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  if (imageUrl) {
    const imgWrap = document.createElement('div');
    imgWrap.className = 'message-image-attachment';
    const img = document.createElement('img');
    img.src = imageUrl;
    img.alt = 'Uploaded Trading Chart';
    img.addEventListener('click', () => window.open(imageUrl, '_blank'));
    imgWrap.appendChild(img);
    bubble.appendChild(imgWrap);
  }

  const textDiv = document.createElement('div');
  textDiv.className = 'message-text';
  if (role === 'assistant') {
    textDiv.innerHTML = marked.parse(content || '');
  } else {
    textDiv.textContent = content;
  }
  bubble.appendChild(textDiv);
  row.appendChild(bubble);

  DOM.chatMessages.appendChild(row);
  scrollToBottom();
  return { row, bubble, textDiv };
}

function scrollToBottom() {
  const container = DOM.mainContentFlow;
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

// ============================================================================
// Form Submission & Live Server-Sent Events (SSE) Streaming
// ============================================================================

async function handleFormSubmit() {
  const messageText = DOM.messageInput.value.trim();
  const imageFile = state.attachedImageFile;

  if (!messageText && !imageFile) return;

  // 1. Render User Message Immediately
  let previewUrl = null;
  if (imageFile) {
    previewUrl = URL.createObjectURL(imageFile);
  }
  appendMessageElement('user', messageText, previewUrl);

  // 2. Prepare Form Data
  const formData = new FormData();
  formData.append('message', messageText || 'Please analyze this trading chart pattern and forecast price targets.');
  if (state.currentConversationId) {
    formData.append('conversation_id', state.currentConversationId);
  }
  if (imageFile) {
    formData.append('image', imageFile);
  }

  // 3. Clear Input Field & Image Attachment
  DOM.messageInput.value = '';
  autoResizeTextarea(DOM.messageInput);
  clearAttachedImage();
  setStreamingState(true);

  // 4. Create Assistant Streaming Bubble
  const assistantBubble = appendMessageElement('assistant', '');
  const streamingTextContainer = assistantBubble.textDiv;
  const cursor = document.createElement('span');
  cursor.className = 'streaming-cursor';
  assistantBubble.bubble.appendChild(cursor);

  let accumulatedResponse = '';

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop(); // Keep last partial chunk

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const jsonStr = line.substring(6).trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'init') {
              if (data.conversation_id) {
                state.currentConversationId = data.conversation_id;
              }
            } else if (data.type === 'token') {
              accumulatedResponse += data.token;
              streamingTextContainer.innerHTML = marked.parse(accumulatedResponse);
              scrollToBottom();
            } else if (data.type === 'error') {
              accumulatedResponse += `\n\n> ⚠️ **${data.error_code || 'Error'}:** ${data.message}`;
              if (data.setup_instructions) {
                accumulatedResponse += `\n\n${data.setup_instructions}`;
              }
              streamingTextContainer.innerHTML = marked.parse(accumulatedResponse);
              scrollToBottom();
            } else if (data.type === 'done') {
              await loadConversations();
            }
          } catch (e) {
            console.error('Failed to parse SSE JSON chunk:', e);
          }
        }
      }
    }
  } catch (err) {
    console.error('Chat error:', err);
    accumulatedResponse += `\n\n> ⚠️ **Connection Error:** Could not reach StockMind AI backend (${err.message}). Ensure server is running on port 8000.`;
    streamingTextContainer.innerHTML = marked.parse(accumulatedResponse);
  } finally {
    if (cursor.parentNode) {
      cursor.parentNode.removeChild(cursor);
    }
    setStreamingState(false);
    DOM.messageInput.focus();
    await loadConversations();
  }
}

function setStreamingState(isStreaming) {
  state.isStreaming = isStreaming;
  DOM.sendBtn.disabled = isStreaming;
  DOM.messageInput.disabled = isStreaming;
  if (!isStreaming) {
    DOM.messageInput.focus();
  }
}
