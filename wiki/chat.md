---
title: Wiki Chat
search:
  exclude: true
---

<style>
.md-content__inner { max-width: 100%; padding: 0 !important; }
.md-typeset h1 { display: none; }

.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 120px);
  max-width: 800px;
  margin: 0 auto;
  padding: 1rem;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.chat-header {
  text-align: center;
  padding: 1.5rem 0;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
  margin-bottom: 1rem;
}

.chat-header h2 {
  margin: 0 0 .3rem;
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--md-default-fg-color);
}

.chat-header p {
  margin: 0;
  font-size: .8rem;
  color: var(--md-default-fg-color--light);
}

.chat-setup {
  text-align: center;
  padding: 2rem;
  background: var(--md-code-bg-color);
  border-radius: 8px;
  margin-bottom: 1rem;
}

.chat-setup label {
  font-size: .85rem;
  color: var(--md-default-fg-color--light);
  display: block;
  margin-bottom: .5rem;
}

.chat-setup input {
  width: 320px;
  max-width: 100%;
  padding: .5rem .8rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 6px;
  font-size: .85rem;
  font-family: monospace;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
}

.chat-setup button {
  display: inline-block;
  margin-top: .8rem;
  padding: .5rem 1.5rem;
  background: var(--md-primary-fg-color);
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: .85rem;
  cursor: pointer;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: .5rem 0;
  display: flex;
  flex-direction: column;
  gap: .8rem;
}

.message {
  padding: .8rem 1rem;
  border-radius: 8px;
  font-size: .88rem;
  line-height: 1.65;
  max-width: 85%;
}

.message-user {
  background: var(--md-primary-fg-color);
  color: #fff;
  align-self: flex-end;
  border-bottom-right-radius: 2px;
}

.message-assistant {
  background: var(--md-code-bg-color);
  color: var(--md-default-fg-color);
  align-self: flex-start;
  border-bottom-left-radius: 2px;
}

.message-assistant code {
  background: var(--md-default-bg-color);
  padding: .1rem .3rem;
  border-radius: 3px;
  font-size: .8rem;
}

.message-assistant pre {
  background: var(--md-default-bg-color);
  padding: .6rem .8rem;
  border-radius: 6px;
  overflow-x: auto;
  margin: .5rem 0;
  font-size: .78rem;
}

.message-loading {
  color: var(--md-default-fg-color--light);
  font-style: italic;
}

.chat-input-area {
  display: flex;
  gap: .5rem;
  padding: 1rem 0 0;
  border-top: 1px solid var(--md-default-fg-color--lightest);
}

.chat-input-area textarea {
  flex: 1;
  padding: .6rem .8rem;
  border: 1px solid var(--md-default-fg-color--lightest);
  border-radius: 8px;
  font-size: .88rem;
  font-family: inherit;
  resize: none;
  background: var(--md-default-bg-color);
  color: var(--md-default-fg-color);
  min-height: 44px;
  max-height: 120px;
}

.chat-input-area textarea:focus {
  outline: none;
  border-color: var(--md-primary-fg-color);
}

.chat-input-area button {
  padding: .6rem 1.2rem;
  background: var(--md-primary-fg-color);
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: .85rem;
  cursor: pointer;
  align-self: flex-end;
}

.chat-input-area button:disabled {
  opacity: .5;
  cursor: not-allowed;
}

.chat-info {
  text-align: center;
  font-size: .7rem;
  color: var(--md-default-fg-color--lightest);
  padding: .5rem 0 0;
}
</style>

<div class="chat-container">
  <div class="chat-header">
    <h2>ML & Forecasting Wiki Chat</h2>
    <p>Fragen an das Wiki stellen. Claude durchsucht alle Wiki-Seiten und antwortet.</p>
  </div>

  <div id="setup" class="chat-setup">
    <label>Anthropic API Key eingeben</label>
    <input type="password" id="api-key" placeholder="sk-ant-...">
    <p style="font-size:.72rem; color:var(--md-default-fg-color--lightest); margin-top:.5rem;">
      Der Key bleibt im Browser und wird nicht gespeichert.
    </p>
    <button onclick="startChat()">Chat starten</button>
  </div>

  <div id="chat-area" style="display:none; flex:1; display:none; flex-direction:column;">
    <div class="chat-messages" id="messages"></div>
    <div class="chat-input-area">
      <textarea id="user-input" placeholder="Frage an das Wiki..." rows="1"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();sendMessage()}"></textarea>
      <button id="send-btn" onclick="sendMessage()">Senden</button>
    </div>
    <div class="chat-info">Antworten basieren auf den Wiki-Inhalten. Kosten: ca. 1-3 Cents pro Frage.</div>
  </div>
</div>

<script>
let apiKey = '';
let wikiContent = '';
let conversationHistory = [];

async function loadWikiBundle() {
  try {
    const response = await fetch('/ml-wiki/assets/wiki-bundle.json');
    const pages = await response.json();
    wikiContent = pages.map(p =>
      `=== ${p.path} ===\n${p.content}`
    ).join('\n\n');
    return true;
  } catch (e) {
    console.error('Wiki-Bundle konnte nicht geladen werden:', e);
    return false;
  }
}

async function startChat() {
  apiKey = document.getElementById('api-key').value.trim();
  if (!apiKey.startsWith('sk-ant-')) {
    alert('Bitte einen gültigen Anthropic API Key eingeben (beginnt mit sk-ant-).');
    return;
  }

  const loaded = await loadWikiBundle();
  if (!loaded) {
    alert('Wiki-Inhalte konnten nicht geladen werden.');
    return;
  }

  document.getElementById('setup').style.display = 'none';
  const chatArea = document.getElementById('chat-area');
  chatArea.style.display = 'flex';
  document.getElementById('user-input').focus();

  addMessage('assistant',
    'Wiki geladen. Ich kenne alle Seiten und kann Fragen beantworten. Was möchtest du wissen?'
  );
}

function addMessage(role, content) {
  const messages = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = `message message-${role}`;

  // Einfaches Markdown-Rendering
  let html = content
    .replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br>');

  div.innerHTML = html;
  messages.appendChild(div);
  messages.scrollTop = messages.scrollHeight;
  return div;
}

async function sendMessage() {
  const input = document.getElementById('user-input');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  input.style.height = 'auto';
  document.getElementById('send-btn').disabled = true;

  addMessage('user', question);
  const loadingDiv = addMessage('assistant', 'Durchsuche Wiki-Seiten...');
  loadingDiv.classList.add('message-loading');

  conversationHistory.push({ role: 'user', content: question });

  try {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'x-api-key': apiKey,
        'anthropic-version': '2023-06-01',
        'anthropic-dangerous-direct-browser-access': 'true'
      },
      body: JSON.stringify({
        model: 'claude-sonnet-4-20250514',
        max_tokens: 2048,
        system: `Du bist ein Assistent für das ML & Forecasting Wiki des BI & ML Teams.
Beantworte Fragen basierend auf den folgenden Wiki-Inhalten. Antworte auf Deutsch.
Wenn die Antwort nicht in den Wiki-Inhalten steht, sage das ehrlich.
Verweise auf relevante Wiki-Seiten wenn möglich.

WIKI-INHALTE:
${wikiContent}`,
        messages: conversationHistory
      })
    });

    const data = await response.json();

    if (data.error) {
      throw new Error(data.error.message);
    }

    const answer = data.content
      .filter(c => c.type === 'text')
      .map(c => c.text)
      .join('\n');

    conversationHistory.push({ role: 'assistant', content: answer });

    loadingDiv.remove();
    addMessage('assistant', answer);

  } catch (error) {
    loadingDiv.remove();
    addMessage('assistant', `Fehler: ${error.message}`);
    conversationHistory.pop();
  }

  document.getElementById('send-btn').disabled = false;
  input.focus();
}

// Textarea auto-resize
document.addEventListener('DOMContentLoaded', () => {
  const textarea = document.getElementById('user-input');
  if (textarea) {
    textarea.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
  }
});
</script>
