// ─────────────────────────────────────────────────
//  Kannada SSL — AI Chatbot
//  Dataset-grounded chatbot: talks to /api/chatbot
// ─────────────────────────────────────────────────

const CHAT_SPEECH_REC_LANG = {
  'en': 'en-IN', 'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN', 'ta': 'ta-IN',
  'ml': 'ml-IN', 'mr': 'mr-IN', 'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
};

const CHAT_SYNTH_LANG = {
  'en': 'en-US', 'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN', 'ta': 'ta-IN',
  'ml': 'ml-IN', 'mr': 'mr-IN', 'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
  'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU', 'ar': 'ar-SA', 'zh-CN': 'zh-CN',
  'ja': 'ja-JP', 'ko': 'ko-KR', 'bn': 'bn-IN', 'gu': 'gu-IN', 'ur': 'ur-IN',
};
Object.assign(CHAT_SPEECH_REC_LANG, {
  'it': 'it-IT', 'pt': 'pt-PT', 'ru': 'ru-RU', 'ar': 'ar-SA', 'zh-CN': 'zh-CN',
  'ja': 'ja-JP', 'ko': 'ko-KR', 'bn': 'bn-IN', 'gu': 'gu-IN', 'ur': 'ur-IN',
});

function chatSpeak(text, langCode) {
  if (!window.speechSynthesis || !text) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = CHAT_SYNTH_LANG[langCode] || 'en-US';
  window.speechSynthesis.speak(utter);
}

let chatSending = false;
let chatRecognition = null;
let chatListening = false;

function appendMessage(role, text, meta, speakLang) {
  const container = document.getElementById('chatMessages');
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;
  if (speakLang) {
    const btn = document.createElement('button');
    btn.className = 'action-btn';
    btn.textContent = '🔊 Listen';
    btn.style.marginTop = '0.4rem';
    btn.onclick = () => chatSpeak(text, speakLang);
    bubble.appendChild(document.createElement('br'));
    bubble.appendChild(btn);
  }
  msg.appendChild(bubble);

  if (meta) {
    const metaEl = document.createElement('div');
    metaEl.className = 'chat-meta';
    metaEl.textContent = meta;
    msg.appendChild(metaEl);
  }

  container.appendChild(msg);
  container.scrollTop = container.scrollHeight;
}

function setTyping(visible) {
  document.getElementById('chatTyping').style.display = visible ? 'flex' : 'none';
  if (visible) {
    const container = document.getElementById('chatMessages');
    container.scrollTop = container.scrollHeight;
  }
}

window.addEventListener('error', (e) => {
  try { appendMessage('bot', 'Page error: ' + e.message); } catch (_) {}
});

async function sendChatMessage() {
  if (chatSending) return;

  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  const targetLang = ((document.getElementById('chatTargetLang') || {}).value || 'kn');
  const sourceLang = ((document.getElementById('chatSourceLang') || {}).value || 'auto');

  appendMessage('user', message);
  input.value = '';
  chatSending = true;
  setTyping(true);

  try {
    const resp = await fetch('/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, target_language: targetLang, source_language: sourceLang }),
    });
    const data = await resp.json();
    setTyping(false);

    if (!data.success) {
      appendMessage('bot', (data.error || 'Something went wrong. Please try again.') +
        (data.hint ? ' ' + data.hint : ''));
      return;
    }

    const names = { en:'English', kn:'Kannada', hi:'Hindi', te:'Telugu', ta:'Tamil', ml:'Malayalam',
      mr:'Marathi', gu:'Gujarati', bn:'Bengali', ur:'Urdu', fr:'French', de:'German', es:'Spanish',
      'zh-CN':'Chinese', ar:'Arabic', ja:'Japanese', ko:'Korean', ru:'Russian', pt:'Portuguese', it:'Italian' };
    const meta = data.note ||
      `${names[data.detected_language] || data.detected_language} → ${names[data.target_language] || data.target_language}`;

    appendMessage('bot', data.reply, meta, data.target_language);
    if (!data.note) chatSpeak(data.reply, data.target_language);
  } catch (e) {
    setTyping(false);
    appendMessage('bot', 'I could not reach the conversation service. Please try again.');
  } finally {
    chatSending = false;
  }
}

async function resetChat() {
  try {
    await fetch('/api/chatbot/reset', { method: 'POST' });
  } catch {}
  const container = document.getElementById('chatMessages');
  container.innerHTML = '';
  appendMessage('bot', 'A fresh conversation is ready. Speak or type to translate.');
}

function toggleChatMic() {
  if (chatListening) return;
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    alert('Speech recognition is not supported in this browser. Try Chrome or Edge.');
    return;
  }

  const sourceLang = ((document.getElementById('chatSourceLang') || {}).value || 'auto');
  const micBtn = document.getElementById('chatMicBtn');

  chatRecognition = new SpeechRec();
  chatRecognition.lang = CHAT_SPEECH_REC_LANG[sourceLang] || 'en-IN';  // 'auto' -> English default
  chatRecognition.interimResults = false;
  chatRecognition.maxAlternatives = 1;

  chatRecognition.onstart = () => {
    chatListening = true;
    micBtn.classList.add('listening');
  };

  chatRecognition.onresult = (event) => {
    const heard = event.results[0][0].transcript;
    document.getElementById('chatInput').value = heard;
    sendChatMessage();
  };

  chatRecognition.onerror = () => { chatListening = false; micBtn.classList.remove('listening'); };
  chatRecognition.onend = () => { chatListening = false; micBtn.classList.remove('listening'); };

  try {
    chatRecognition.start();
  } catch (e) {
    chatListening = false;
    micBtn.classList.remove('listening');
  }
}
