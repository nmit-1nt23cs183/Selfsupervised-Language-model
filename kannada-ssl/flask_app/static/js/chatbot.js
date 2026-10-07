// ─────────────────────────────────────────────────
//  Kannada SSL — AI Chatbot
//  Dataset-grounded chatbot: talks to /api/chatbot
// ─────────────────────────────────────────────────

const CHAT_SPEECH_REC_LANG = {
  'en': 'en-IN', 'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN', 'ta': 'ta-IN',
  'ml': 'ml-IN', 'mr': 'mr-IN', 'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
};

let chatSending = false;
let chatRecognition = null;
let chatListening = false;
let sampleDataNoticeShown = false;

function appendMessage(role, text, meta) {
  const container = document.getElementById('chatMessages');
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;

  const bubble = document.createElement('div');
  bubble.className = 'chat-bubble';
  bubble.textContent = text;
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

async function sendChatMessage() {
  if (chatSending) return;

  const input = document.getElementById('chatInput');
  const message = input.value.trim();
  if (!message) return;

  const targetLang = document.getElementById('chatTargetLang').value;

  appendMessage('user', message);
  input.value = '';
  chatSending = true;
  setTyping(true);

  try {
    const resp = await fetch('/api/chatbot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, target_language: targetLang }),
    });
    const data = await resp.json();
    setTyping(false);

    if (!data.success) {
      appendMessage('bot', data.error || 'Something went wrong. Please try again.');
      return;
    }

    let meta = data.grounded_in_dataset
      ? 'A close match to your question'
      : 'I’m still learning this question';
    if (data.predicted_topic) meta += ` · ${data.predicted_topic}`;

    if (data.using_sample_data && !sampleDataNoticeShown) {
      sampleDataNoticeShown = true;
      appendMessage(
        'bot',
         'I have a smaller set of examples available right now, so my answer may be brief. ' +
         'You can still ask me about Kannada words and everyday topics.'
      );
    }

    appendMessage('bot', data.reply, meta);
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
  appendMessage('bot', 'A fresh conversation is ready. What would you like to know?');
}

function toggleChatMic() {
  if (chatListening) return;
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    alert('Speech recognition is not supported in this browser. Try Chrome or Edge.');
    return;
  }

  const targetLang = document.getElementById('chatTargetLang').value;
  const micBtn = document.getElementById('chatMicBtn');

  chatRecognition = new SpeechRec();
  chatRecognition.lang = CHAT_SPEECH_REC_LANG[targetLang] || 'en-IN';
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
