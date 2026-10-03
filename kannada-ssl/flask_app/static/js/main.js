// ─────────────────────────────────────────────────
//  Kannada SSL — Translator
//  Features: text translation, speech input (mic),
//             text-to-speech output, virtual keyboard
// ─────────────────────────────────────────────────

/* ── Speech synthesis lang map ────────────────── */
const SPEECH_SYNTH_LANG = {
  'en': 'en-US', 'en-US': 'en-US', 'en-IN': 'en-IN',
  'kn': 'kn-IN', 'hi': 'hi-IN', 'te': 'te-IN',
  'ta': 'ta-IN', 'ml': 'ml-IN', 'mr': 'mr-IN',
  'fr': 'fr-FR', 'de': 'de-DE', 'es': 'es-ES',
  'it': 'it-IT', 'pt': 'pt-BR', 'ru': 'ru-RU',
  'ar': 'ar-SA', 'zh-CN': 'zh-CN', 'zh': 'zh-CN',
  'ja': 'ja-JP', 'ko': 'ko-KR', 'bn': 'bn-IN',
  'gu': 'gu-IN', 'pa': 'pa-IN', 'or': 'or-IN', 'as': 'as-IN',
  'ne': 'ne-NP', 'sa': 'sa-IN', 'ur': 'ur-IN',
};

/* ── State ──────────────────────────────────────── */
let kbdVisible  = false;
let currentKbd  = 'kn';
let recognition = null;
let isListening = false;
let isSpeaking  = false;

/* ════════════════════════════════════════════════
   SPEECH INPUT (microphone)
   ════════════════════════════════════════════════ */

function toggleMic() {
  if (isListening) { stopMic(); } else { startMic(); }
}

function startMic() {
  const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRec) {
    showMicError('Speech recognition not supported in this browser. Please use Chrome or Edge.');
    return;
  }

  const langCode = document.getElementById('speechLang').value;

  recognition           = new SpeechRec();
  recognition.lang      = langCode;
  recognition.continuous      = false;
  recognition.interimResults  = true;
  recognition.maxAlternatives = 1;

  const ta         = document.getElementById('inputText');
  const micBtn     = document.getElementById('micBtn');
  const micLabel   = document.getElementById('micLabel');
  const micStatus  = document.getElementById('micStatus');
  const micText    = document.getElementById('micStatusText');

  recognition.onstart = () => {
    isListening = true;
    micBtn.classList.add('listening');
    micLabel.textContent = '● Listening…';
    micStatus.style.display = 'flex';
    micText.textContent = 'Listening — speak now';
    // Clear previous text so the new speech goes in cleanly
    if (ta.value.trim() === '') ta.value = '';
  };

  let finalTranscript = '';

  recognition.onresult = (event) => {
    let interim = '';
    finalTranscript = '';
    for (let i = 0; i < event.results.length; i++) {
      const t = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += t;
      } else {
        interim += t;
      }
    }
    // Show interim in status bar
    if (interim) micText.textContent = interim;

    // When we have a final result, put it in the textarea
    if (finalTranscript) {
      ta.value = finalTranscript.trim();
      onInputChange();
      micText.textContent = '✓ Heard: "' + finalTranscript.trim() + '"';
    }
  };

  recognition.onerror = (event) => {
    const msgs = {
      'no-speech':         'No speech detected — try speaking louder or closer to the mic.',
      'audio-capture':     'Microphone not found. Check your mic and browser permissions.',
      'not-allowed':       'Microphone access denied. Please allow microphone in browser settings.',
      'network':           'Network error during speech recognition.',
      'aborted':           'Listening stopped.',
    };
    micText.textContent = msgs[event.error] || ('Error: ' + event.error);
    stopMic(false);
  };

  recognition.onend = () => {
    stopMic(false);
    // Auto-translate if we got something
    if (finalTranscript.trim()) {
      setTimeout(() => {
        document.getElementById('micStatus').style.display = 'none';
        runTranslation();
      }, 600);
    }
  };

  try {
    recognition.start();
  } catch(e) {
    showMicError('Could not start microphone: ' + e.message);
  }
}

function stopMic(abort = true) {
  isListening = false;
  if (recognition) {
    try { if (abort) recognition.abort(); else recognition.stop(); } catch {}
    recognition = null;
  }
  const micBtn   = document.getElementById('micBtn');
  const micLabel = document.getElementById('micLabel');
  micBtn.classList.remove('listening');
  micLabel.textContent = 'Speak';
}

function showMicError(msg) {
  const micStatus = document.getElementById('micStatus');
  const micText   = document.getElementById('micStatusText');
  micStatus.style.display = 'flex';
  micText.textContent = msg;
  setTimeout(() => { micStatus.style.display = 'none'; }, 5000);
}

/* ════════════════════════════════════════════════
   TEXT-TO-SPEECH OUTPUT (speak translation aloud)
   ════════════════════════════════════════════════ */

function speakOutput() {
  const outputEl = document.getElementById('outputArea').querySelector('.output-text');
  if (!outputEl || !outputEl.textContent.trim()) {
    alert('No output to speak yet. Translate something first.');
    return;
  }

  if (!window.speechSynthesis) {
    alert('Text-to-speech is not supported in this browser.');
    return;
  }

  // Stop any current speech
  window.speechSynthesis.cancel();

  const text       = outputEl.textContent.trim();
  const outputLang = document.getElementById('outputLang').value;
  const speechLang = SPEECH_SYNTH_LANG[outputLang] || outputLang;

  const utterance      = new SpeechSynthesisUtterance(text);
  utterance.lang       = speechLang;
  utterance.rate       = 0.92;
  utterance.pitch      = 1.0;
  utterance.volume     = 1.0;

  // Try to pick a matching voice
  const voices = window.speechSynthesis.getVoices();
  const match  = voices.find(v => v.lang.startsWith(speechLang.split('-')[0]));
  if (match) utterance.voice = match;

  utterance.onstart = () => {
    isSpeaking = true;
    const listenBtn  = document.getElementById('listenBtn');
    const stopBtn    = document.getElementById('stopSpeakBtn');
    listenBtn.style.display = 'none';
    stopBtn.style.display   = 'flex';
  };

  utterance.onend = utterance.onerror = () => {
    isSpeaking = false;
    const listenBtn = document.getElementById('listenBtn');
    const stopBtn   = document.getElementById('stopSpeakBtn');
    listenBtn.style.display = 'flex';
    stopBtn.style.display   = 'none';
  };

  window.speechSynthesis.speak(utterance);
}

function stopSpeaking() {
  window.speechSynthesis?.cancel();
  isSpeaking = false;
  const listenBtn = document.getElementById('listenBtn');
  const stopBtn   = document.getElementById('stopSpeakBtn');
  if (listenBtn) listenBtn.style.display = 'flex';
  if (stopBtn)   stopBtn.style.display   = 'none';
}

/* ════════════════════════════════════════════════
   TRANSLATION
   ════════════════════════════════════════════════ */

async function runTranslation() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) { document.getElementById('inputText').focus(); return; }

  const targetLang = document.getElementById('outputLang').value;
  const btn  = document.getElementById('translateBtn');
  const area = document.getElementById('outputArea');

  btn.disabled = true;
  stopSpeaking();
  area.innerHTML = `<div class="loading-spinner"><div class="spinner"></div> Translating…</div>`;
  document.getElementById('outputActions').style.display = 'none';

  try {
    const resp = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, target_language: targetLang }),
    });
    const data = await resp.json();

    if (!data.success) {
      const isInstall = data.error && data.error.includes('not installed');
      area.innerHTML = isInstall ? `
        <div class="setup-guide">
          <div class="sg-title">Translation is not ready yet</div>
          <p>This language service needs a little attention before it can translate. Please try again in a moment.</p>
        </div>` :
        `<div style="color:var(--danger);padding:1rem">${data.error || 'Translation failed'}${data.hint ? '<br><small style="color:var(--text3)">'+data.hint+'</small>':''}</div>`;
      return;
    }

    area.innerHTML = `<div class="output-text">${data.final_output || '—'}</div>`;
    document.getElementById('outputActions').style.display = 'flex';

    // Update detected lang badge
    if (data.detected_language) {
      const names = {
        en:'English',kn:'Kannada',hi:'Hindi',te:'Telugu',ta:'Tamil',
        ml:'Malayalam',mr:'Marathi',fr:'French',de:'German',es:'Spanish',
        ar:'Arabic','zh-CN':'Chinese',zh:'Chinese',ja:'Japanese',ko:'Korean',
         ru:'Russian',pt:'Portuguese',it:'Italian',bn:'Bengali',gu:'Gujarati',
         pa:'Punjabi',or:'Odia',ur:'Urdu',as:'Assamese',ne:'Nepali',sa:'Sanskrit'
      };
      document.getElementById('detectedLang').textContent =
        names[data.detected_language] || data.detected_language.toUpperCase();
    }

    // Pipeline steps
    if (data.steps?.length) {
      const det  = document.getElementById('pipelineDetails');
      const list = document.getElementById('pipelineStepsList');
      det.style.display = 'block';
      list.innerHTML = data.steps.map(s => `
        <div class="ps-item">
          <div class="ps-num">${s.step}</div>
          <div>
            <div class="ps-name">${s.name}</div>
            <div class="ps-out">${s.output}</div>
          </div>
        </div>`).join('');
    }

    // Auto-speak if coming from microphone
    if (document.getElementById('micStatus').style.display === 'none' &&
        document.getElementById('micLabel').textContent.includes('Speak')) {
      // Don't auto-speak for typed input; only do it on demand
    }

  } catch(e) {
    area.innerHTML = `<div style="color:var(--danger);padding:1rem">
      Network error: ${e.message}<br>
      <small style="color:var(--text3)">Make sure the Flask server is running: python app.py</small>
    </div>`;
  } finally {
    btn.disabled = false;
  }
}

/* ════════════════════════════════════════════════
   HELPERS
   ════════════════════════════════════════════════ */

function onInputChange() {
  const val = document.getElementById('inputText').value;
  document.getElementById('clearBtn').style.display = val ? 'block' : 'none';
}

function clearInput() {
  document.getElementById('inputText').value = '';
  document.getElementById('detectedLang').textContent = 'Auto-detect';
  document.getElementById('clearBtn').style.display = 'none';
  resetOutput();
}

function resetOutput() {
  document.getElementById('outputArea').innerHTML = `
    <div class="output-placeholder">
         <div class="placeholder-kannada">ಅನುವಾದ</div>
         <p>Your translation will appear here</p>
         <p style="font-size:0.78rem;margin-top:0.25rem">Press Ctrl+Enter or choose Translate</p>
    </div>`;
  document.getElementById('outputActions').style.display = 'none';
  document.getElementById('pipelineDetails').style.display = 'none';
  stopSpeaking();
}

function setOutputLang(code) {
  document.getElementById('outputLang').value = code;
  document.querySelectorAll('.ql-btn').forEach(b => {
    b.classList.toggle('active', b.getAttribute('onclick') === `setOutputLang('${code}')`);
  });
}

function onOutputLangChange() {
  // If currently speaking, re-speak in new language
  if (isSpeaking) { stopSpeaking(); setTimeout(speakOutput, 300); }
}

function copyOutput() {
  const text = document.getElementById('outputArea').querySelector('.output-text')?.textContent || '';
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById('copyBtn');
    const orig = btn.innerHTML;
    btn.textContent = '✓ Copied!';
    btn.style.color = 'var(--success)';
    setTimeout(() => { btn.innerHTML = orig; btn.style.color = ''; }, 1800);
  });
}

function setExample(text) {
  document.getElementById('inputText').value = text;
  onInputChange();
}

/* ════════════════════════════════════════════════
   VIRTUAL KEYBOARD
   ════════════════════════════════════════════════ */

const KEYBOARDS = {
  kn: {
    rows: [
      ['ಅ','ಆ','ಇ','ಈ','ಉ','ಊ','ಋ','ಎ','ಏ','ಐ','ಒ','ಓ','ಔ','ಂ','ಃ'],
      ['ಕ','ಖ','ಗ','ಘ','ಙ','ಚ','ಛ','ಜ','ಝ','ಞ','ಟ','ಠ','ಡ','ಢ','ಣ'],
      ['ತ','ಥ','ದ','ಧ','ನ','ಪ','ಫ','ಬ','ಭ','ಮ','ಯ','ರ','ಲ','ವ','ಶ'],
      ['ಷ','ಸ','ಹ','ಳ','ಕ್ಷ','ಜ್ಞ','।','೦','೧','೨','೩','೪','೫','೬','೭'],
      ['ಾ','ಿ','ೀ','ು','ೂ','ೃ','ೆ','ೇ','ೈ','ೊ','ೋ','ೌ','್','಼','ಽ'],
    ],
  },
  en: {
    rows: [
      ['q','w','e','r','t','y','u','i','o','p'],
      ['a','s','d','f','g','h','j','k','l'],
      ['z','x','c','v','b','n','m'],
      ['Q','W','E','R','T','Y','U','I','O','P'],
      ['1','2','3','4','5','6','7','8','9','0',',','.','!','?','\''],
    ],
  },
  hi: {
    rows: [
      ['अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ','अं','अः'],
      ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण'],
      ['त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श'],
      ['ष','स','ह','क्ष','ज्ञ','त्र','।','०','१','२','३','४','५'],
      ['ा','ि','ी','ु','ू','ृ','े','ै','ो','ौ','्','ं','ः'],
    ],
  },
  te: {
    rows: [
      ['అ','ఆ','ఇ','ఈ','ఉ','ఊ','ఋ','ఎ','ఏ','ఐ','ఒ','ఓ','ఔ'],
      ['క','ఖ','గ','ఘ','చ','ఛ','జ','ఝ','ట','ఠ','డ','ఢ','ణ','త','థ'],
      ['ద','ధ','న','ప','ఫ','బ','భ','మ','య','ర','ల','వ','శ','ష','స'],
      ['హ','ళ','క్ష','జ్ఞ','౦','౧','౨','౩','౪','౫','౬','౭','।'],
      ['ా','ి','ీ','ు','ూ','ృ','ె','ే','ై','ొ','ో','ౌ','్'],
    ],
  }
};

/* Script keyboards are a typing aid and stay independent from translation
   language availability on the server. */
Object.assign(KEYBOARDS, {
  ta: { rows: [
    ['அ','ஆ','இ','ஈ','உ','ஊ','எ','ஏ','ஐ','ஒ','ஓ','ஔ','ஂ','ஃ'],
    ['க','ங','ச','ஞ','ட','ண','த','ந','ப','ம','ய','ர','ல','வ'],
    ['ழ','ள','ற','ன','ஜ','ஷ','ஸ','ஹ','ஶ','க்ஷ','ஸ்ரீ','।','ௐ'],
    ['ா','ி','ீ','ு','ூ','ெ','ே','ை','ொ','ோ','ௌ','்','ௗ','ஂ'],
  ]},
  ml: { rows: [
    ['അ','ആ','ഇ','ഈ','ഉ','ഊ','ഋ','എ','ഏ','ഐ','ഒ','ഓ','ഔ','ം','ഃ'],
    ['ക','ഖ','ഗ','ഘ','ങ','ച','ഛ','ജ','ഝ','ഞ','ട','ഠ','ഡ','ഢ','ണ'],
    ['ത','ഥ','ദ','ധ','ന','പ','ഫ','ബ','ഭ','മ','യ','ര','ല','വ','ശ'],
    ['ഷ','സ','ഹ','ള','ഴ','റ','ക്ഷ','ജ്ഞ','൦','൧','൨','൩','।'],
    ['ാ','ി','ീ','ു','ൂ','ൃ','െ','േ','ൈ','ൊ','ോ','ൌ','്','ം'],
  ]},
  mr: { rows: [
    ['अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ','अं','अः'],
    ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण'],
    ['त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श'],
    ['ष','स','ह','ळ','क्ष','ज्ञ','त्र','द्व','।','०','१','२','३','४'],
    ['ा','ि','ी','ु','ू','ृ','े','ै','ो','ौ','्','ं','ः'],
  ]},
  bn: { rows: [
    ['অ','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ং','ঃ'],
    ['ক','খ','গ','ঘ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ঠ','ড','ঢ','ণ'],
    ['ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','র','ল','শ'],
    ['ষ','স','হ','ড়','ঢ়','য়','ক্ষ','জ্ঞ','।','০','১','২','৩','৪'],
    ['া','ি','ী','ু','ূ','ৃ','ে','ৈ','ো','ৌ','্','ঁ'],
  ]},
  gu: { rows: [
    ['અ','આ','ઇ','ઈ','ઉ','ઊ','ઋ','એ','ઐ','ઓ','ઔ','અં','અઃ'],
    ['ક','ખ','ગ','ઘ','ઙ','ચ','છ','જ','ઝ','ઞ','ટ','ઠ','ડ','ઢ','ણ'],
    ['ત','થ','દ','ધ','ન','પ','ફ','બ','ભ','મ','ય','ર','લ','વ','શ'],
    ['ષ','સ','હ','ળ','ક્ષ','જ્ઞ','।','૦','૧','૨','૩','૪','૫','૬'],
    ['ા','િ','ી','ુ','ૂ','ૃ','ે','ૈ','ો','ૌ','્','ં','ઃ'],
  ]},
  pa: { rows: [
    ['ਅ','ਆ','ਇ','ਈ','ਉ','ਊ','ਏ','ਐ','ਓ','ਔ','ਅੰ','ਅੱ'],
    ['ਕ','ਖ','ਗ','ਘ','ਙ','ਚ','ਛ','ਜ','ਝ','ਞ','ਟ','ਠ','ਡ','ਢ','ਣ'],
    ['ਤ','ਥ','ਦ','ਧ','ਨ','ਪ','ਫ','ਬ','ਭ','ਮ','ਯ','ਰ','ਲ','ਵ','ੜ'],
    ['ਸ','ਹ','ਖ਼','ਗ਼','ਜ਼','ਫ਼','ਸ਼','ਲ਼','।','੦','੧','੨','੩','੪'],
    ['ਾ','ਿ','ੀ','ੁ','ੂ','ੇ','ੈ','ੋ','ੌ','੍','ੰ','ੱ','ਂ'],
  ]},
  or: { rows: [
    ['ଅ','ଆ','ଇ','ଈ','ଉ','ଊ','ଋ','ଏ','ଐ','ଓ','ଔ','ଂ','ଃ'],
    ['କ','ଖ','ଗ','ଘ','ଙ','ଚ','ଛ','ଜ','ଝ','ଞ','ଟ','ଠ','ଡ','ଢ','ଣ'],
    ['ତ','ଥ','ଦ','ଧ','ନ','ପ','ଫ','ବ','ଭ','ମ','ଯ','ର','ଲ','ଳ'],
    ['ୱ','ଶ','ଷ','ସ','ହ','କ୍ଷ','ଜ୍ଞ','।','୦','୧','୨','୩','୪'],
    ['ା','ି','ୀ','ୁ','ୂ','ୃ','େ','ୈ','ୋ','ୌ','୍','ଂ','ଃ'],
  ]},
  ur: { rows: [
    ['ا','آ','أ','ء','ب','پ','ت','ٹ','ث','ج','چ','ح','خ'],
    ['د','ڈ','ذ','ر','ڑ','ز','ژ','س','ش','ص','ض','ط','ظ'],
    ['ع','غ','ف','ق','ک','گ','ل','م','ن','ں','و','ہ','ی'],
    ['ے','ئ','ؤ','ۓ','۔','،','؟','۰','۱','۲','۳','۴','۵'],
    ['َ','ِ','ُ','ّ','ْ','ٰ','ً','ٍ','ٌ'],
  ]},
  as: { rows: [
    ['অ','আ','ই','ঈ','উ','ঊ','ঋ','এ','ঐ','ও','ঔ','ং','ঃ'],
    ['ক','খ','গ','ঘ','ঙ','চ','ছ','জ','ঝ','ঞ','ট','ঠ','ড','ঢ','ণ'],
    ['ত','থ','দ','ধ','ন','প','ফ','ব','ভ','ম','য','ৰ','ল','শ'],
    ['ষ','স','হ','ড়','ঢ়','য়','ক্ষ','জ্ঞ','।','০','১','২','৩','৪'],
    ['া','ি','ী','ু','ূ','ৃ','ে','ৈ','ো','ৌ','্','ঁ'],
  ]},
  ne: { rows: [
    ['अ','आ','इ','ई','उ','ऊ','ऋ','ए','ऐ','ओ','औ','अं','अः'],
    ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण'],
    ['त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श'],
    ['ष','स','ह','क्ष','ज्ञ','त्र','।','०','१','२','३','४','५','६'],
    ['ा','ि','ी','ु','ू','ृ','े','ै','ो','ौ','्','ं','ः'],
  ]},
  sa: { rows: [
    ['अ','आ','इ','ई','उ','ऊ','ऋ','ॠ','ऌ','ए','ऐ','ओ','औ','अं','अः'],
    ['क','ख','ग','घ','ङ','च','छ','ज','झ','ञ','ट','ठ','ड','ढ','ण'],
    ['त','थ','द','ध','न','प','फ','ब','भ','म','य','र','ल','व','श'],
    ['ष','स','ह','क्ष','ज्ञ','त्र','श्र','।','०','१','२','३','४','५'],
    ['ा','ि','ी','ु','ू','ृ','ॄ','े','ै','ो','ौ','्','ं','ः'],
  ]},
});

function buildKeyboard(lang) {
  const kb = KEYBOARDS[lang];
  if (!kb) return;
  const container = document.getElementById('virtualKeyboard');
  container.innerHTML = '';

  kb.rows.forEach((row, ri) => {
    const rowEl = document.createElement('div');
    rowEl.className = 'kbd-row';
    row.forEach(char => {
      const btn = document.createElement('button');
      btn.className = 'kbd-key';
      btn.textContent = char;
      btn.addEventListener('mousedown', e => { e.preventDefault(); insertChar(char); });
      rowEl.appendChild(btn);
    });
    container.appendChild(rowEl);
    if (ri === 0) {
      const div = document.createElement('div');
      div.className = 'kbd-divider';
      container.appendChild(div);
    }
  });

  // Special keys
  const specRow = document.createElement('div');
  specRow.className = 'kbd-row';
  [
    { label: 'Space',  value: ' ',    cls: 'space' },
    { label: '⌫',      value: 'BKSP', cls: 'backspace' },
    { label: 'Clear',  value: 'CLR',  cls: 'wide' },
  ].forEach(s => {
    const btn = document.createElement('button');
    btn.className = `kbd-key ${s.cls}`;
    btn.textContent = s.label;
    btn.addEventListener('mousedown', e => { e.preventDefault(); handleSpecialKey(s.value); });
    specRow.appendChild(btn);
  });
  container.appendChild(specRow);
}

function insertChar(char) {
  const ta = document.getElementById('inputText');
  const s = ta.selectionStart, e = ta.selectionEnd;
  ta.value = ta.value.slice(0, s) + char + ta.value.slice(e);
  ta.selectionStart = ta.selectionEnd = s + char.length;
  ta.focus(); onInputChange();
}

function handleSpecialKey(val) {
  const ta = document.getElementById('inputText');
  if (val === ' ')    { insertChar(' '); return; }
  if (val === 'BKSP') {
    const s = ta.selectionStart;
    if (s > 0) { ta.value = ta.value.slice(0, s-1) + ta.value.slice(s); ta.selectionStart = ta.selectionEnd = s-1; }
    ta.focus(); onInputChange(); return;
  }
  if (val === 'CLR')  { clearInput(); }
}

function toggleKeyboard() {
  kbdVisible = !kbdVisible;
  document.getElementById('virtualKeyboard').style.display  = kbdVisible ? 'flex' : 'none';
  document.getElementById('kbdLangTabs').style.display      = kbdVisible ? 'flex' : 'none';
  document.getElementById('kbdToggleBtn').classList.toggle('active', kbdVisible);
  if (kbdVisible) buildKeyboard(currentKbd);
}

function switchKbd(lang, btn) {
  currentKbd = lang;
  document.querySelectorAll('.kbd-tab').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  buildKeyboard(lang);
}

/* ════════════════════════════════════════════════
   INIT
   ════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  const navToggle = document.getElementById('navToggle');
  const primaryNavigation = document.getElementById('primaryNavigation');
  if (navToggle && primaryNavigation) {
    navToggle.addEventListener('click', () => {
      const isOpen = primaryNavigation.classList.toggle('is-open');
      navToggle.setAttribute('aria-expanded', String(isOpen));
      navToggle.setAttribute('aria-label', isOpen ? 'Close navigation' : 'Open navigation');
    });

    primaryNavigation.querySelectorAll('a').forEach((link) => {
      link.addEventListener('click', () => {
        primaryNavigation.classList.remove('is-open');
        navToggle.setAttribute('aria-expanded', 'false');
        navToggle.setAttribute('aria-label', 'Open navigation');
      });
    });
  }

  // Ctrl+Enter to translate
  const ta = document.getElementById('inputText');
  if (ta) {
    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); runTranslation(); }
    });
  }

  // Load voices list (needed for TTS on some browsers)
  if (window.speechSynthesis) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
  }

  if (document.getElementById('inputText')) onInputChange();
});
