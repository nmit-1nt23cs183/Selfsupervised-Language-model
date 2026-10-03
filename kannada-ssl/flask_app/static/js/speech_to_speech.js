// ─────────────────────────────────────────────────
//  Kannada SSL — Speech to Speech
//  Quick Mode: browser mic -> /api/translate -> browser TTS
//  File Mode:  record/upload audio -> /api/speech-to-speech -> server audio
// ─────────────────────────────────────────────────

const SPEECH_SYNTH_LANG_S2S = {
  'en': 'en-US', 'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN', 'ta': 'ta-IN',
  'ml': 'ml-IN', 'mr': 'mr-IN', 'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
  'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU', 'ar': 'ar-SA', 'zh-CN': 'zh-CN',
  'ja': 'ja-JP', 'ko': 'ko-KR', 'bn': 'bn-IN', 'gu': 'gu-IN', 'ur': 'ur-IN',
};

const SPEECH_REC_LANG_S2S = {
  'en': 'en-IN', 'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN', 'ta': 'ta-IN',
  'ml': 'ml-IN', 'mr': 'mr-IN', 'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
  'it': 'it-IT', 'pt': 'pt-PT', 'ru': 'ru-RU', 'ar': 'ar-SA', 'zh-CN': 'zh-CN',
  'ja': 'ja-JP', 'ko': 'ko-KR', 'bn': 'bn-IN', 'gu': 'gu-IN', 'ur': 'ur-IN',
};

/* ════════════════════ QUICK MODE ════════════════════ */

let quickRecognition = null;
let quickListening = false;

function toggleQuickMic() {
  if (quickListening) return; // recognition auto-stops after one utterance
  startQuickMic();
}

function startQuickMic() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    alert('Speech recognition is not supported in this browser. Try Chrome or Edge.');
    return;
  }

  const sourceLang = document.getElementById('quickSourceLang').value;
  const targetLang = document.getElementById('quickTargetLang').value;
  const btn = document.getElementById('quickMicBtn');
  const label = document.getElementById('quickMicLabel');

  quickRecognition = new SpeechRec();
  quickRecognition.lang = SPEECH_REC_LANG_S2S[sourceLang] || 'en-IN';
  quickRecognition.interimResults = false;
  quickRecognition.maxAlternatives = 1;

  quickRecognition.onstart = () => {
    quickListening = true;
    btn.classList.add('listening');
    label.textContent = '● Listening…';
  };

  quickRecognition.onresult = async (event) => {
    const heard = event.results[0][0].transcript;
    document.getElementById('quickResult').style.display = 'block';
    document.getElementById('quickHeard').textContent = heard;
    document.getElementById('quickTranslated').textContent = 'Translating…';

    try {
      const resp = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: heard, target_language: targetLang }),
      });
      const data = await resp.json();
      const output = data.final_output || data.error || 'Translation failed';
      document.getElementById('quickTranslated').textContent = output;

      if (data.success) speakText(output, targetLang);
    } catch (e) {
      document.getElementById('quickTranslated').textContent = 'Error: ' + e.message;
    }
  };

  quickRecognition.onerror = () => stopQuickMic();
  quickRecognition.onend = () => stopQuickMic();

  try {
    quickRecognition.start();
  } catch (e) {
    stopQuickMic();
  }
}

function stopQuickMic() {
  quickListening = false;
  const btn = document.getElementById('quickMicBtn');
  const label = document.getElementById('quickMicLabel');
  btn.classList.remove('listening');
  label.textContent = 'Speak once more';
  quickRecognition = null;
}

function speakText(text, langCode) {
  if (!window.speechSynthesis || !text) return;
  window.speechSynthesis.cancel();
  const utter = new SpeechSynthesisUtterance(text);
  utter.lang = SPEECH_SYNTH_LANG_S2S[langCode] || 'en-US';
  window.speechSynthesis.speak(utter);
}

/* ════════════════════ FILE MODE ════════════════════ */

let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let selectedAudioBlob = null;
let selectedAudioName = 'recording.webm';

async function toggleRecording() {
  if (isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      selectedAudioBlob = new Blob(recordedChunks, { type: 'audio/webm' });
      selectedAudioName = 'recording.webm';
      document.getElementById('fileStatus').textContent =
        `Recorded ${(selectedAudioBlob.size / 1024).toFixed(0)} KB`;
      document.getElementById('s2sSubmitBtn').disabled = false;
      stream.getTracks().forEach((t) => t.stop());
    };

    mediaRecorder.start();
    isRecording = true;
    document.getElementById('recordLabel').textContent = 'Stop';
    document.getElementById('recordBtn').classList.add('listening');
  } catch (e) {
    alert('Could not access microphone: ' + e.message);
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    document.getElementById('recordLabel').textContent = 'Record';
    document.getElementById('recordBtn').classList.remove('listening');
  }
}

function onAudioFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;
  selectedAudioBlob = file;
  selectedAudioName = file.name;
    document.getElementById('fileStatus').textContent = `Selected: ${file.name}`;
  document.getElementById('s2sSubmitBtn').disabled = false;
}

async function runSpeechToSpeech() {
  if (!selectedAudioBlob) return;

  const sourceLang = document.getElementById('fileSourceLang').value;
  const targetLang = document.getElementById('fileTargetLang').value;

  document.getElementById('s2sLoading').style.display = 'flex';
  document.getElementById('s2sOutput').style.display = 'none';
  document.getElementById('s2sError').style.display = 'none';
  document.getElementById('s2sSubmitBtn').disabled = true;

  const formData = new FormData();
  formData.append('audio', selectedAudioBlob, selectedAudioName);
  formData.append('source_language', sourceLang);
  formData.append('target_language', targetLang);

  try {
    const resp = await fetch('/api/speech-to-speech', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!data.success) {
      throw new Error(data.error || 'We could not translate this audio.');
    }

    document.getElementById('s2sTranscribed').textContent = data.transcribed_text;
    document.getElementById('s2sTranslated').textContent = data.translated_text;
    const player = document.getElementById('s2sAudioPlayer');
    player.src = data.audio_url;
    document.getElementById('s2sDownloadLink').href = data.audio_url;
    document.getElementById('s2sOutput').style.display = 'block';
    player.play().catch(() => {});
  } catch (e) {
    const errEl = document.getElementById('s2sError');
    errEl.textContent = e.message;
    errEl.style.display = 'block';
  } finally {
    document.getElementById('s2sLoading').style.display = 'none';
    document.getElementById('s2sSubmitBtn').disabled = false;
  }
}
