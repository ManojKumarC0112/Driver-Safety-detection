/**
 * Driver Safety AI — Web Cockpit Dashboard Controller
 * Handles real-time telemetry polling, animated DVI ring gauge, EAR Chart,
 * Dhaba list rendering, and 15-minute power nap alarm timer.
 */

// Chart.js EAR Oscilloscope setup
let earChart = null;
const earHistory = Array(50).fill(0.35);

function initEarChart() {
    const ctx = document.getElementById('earChart').getContext('2d');
    earChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array(50).fill(''),
            datasets: [{
                label: 'EAR',
                data: earHistory,
                borderColor: '#10b981',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true,
                backgroundColor: 'rgba(16, 185, 129, 0.08)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: {
                    min: 0.0,
                    max: 0.45,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#64748b', font: { size: 10 } }
                }
            }
        }
    });
}

// Speech Synthesis Voice Pre-loader & Audio Management
const VOICE_TEXT = {
    english: {
        VOICE_LEVEL_1: "Please stay alert.",
        VOICE_LEVEL_2: "You appear drowsy. Please stay alert.",
        VOICE_LEVEL_3: "Warning. Drowsiness is continuing. Please take a safe break.",
        ALERTNESS_RESTORED: "Alertness restored."
    },
    hindi: {
        VOICE_LEVEL_1: "Kripya savdhaan rahen.",
        VOICE_LEVEL_2: "Aap neend mein lag rahe hain. Kripya savdhaan rahen.",
        VOICE_LEVEL_3: "Chetaavani. Neend jaisi sthiti jaari hai. Kripya safe break lijiye.",
        ALERTNESS_RESTORED: "Alertness wapas aa gayi hai."
    },
    hinglish: {
        VOICE_LEVEL_1: "Please alert rahen.",
        VOICE_LEVEL_2: "Aap drowsy lag rahe hain. Please alert rahen.",
        VOICE_LEVEL_3: "Warning. Drowsiness continue ho rahi hai. Please safe break lein.",
        ALERTNESS_RESTORED: "Alertness restore ho gayi hai."
    }
};

const VOICE_LANGUAGE_META = {
    english: { code: "en-IN", label: "English" },
    hindi: { code: "hi-IN", label: "Hindi" },
    hinglish: { code: "en-IN", label: "Hinglish" }
};

const VOICE_TEST_PROMPTS = {
    english: "Emergency sound test active. Audio alarm is fully operational.",
    hindi: "आपातकालीन साउंड टेस्ट सक्रिय है. ऑडियो अलार्म ठीक से काम कर रहा है.",
    hinglish: "Emergency sound test active hai. Audio alarm bilkul theek chal raha hai."
};

let currentVoiceLanguage = localStorage.getItem('voiceLanguage') || 'english';
let lastVoiceEvent = null;
let lastDhabaVoicePrompt = '';
let isDhabaDrawerAutoOpened = false;
let availableVoices = [];
function loadVoices() {
    if ('speechSynthesis' in window) {
        availableVoices = window.speechSynthesis.getVoices();
    }
}
if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();
}

function getCurrentVoiceLanguageCode() {
    return (VOICE_LANGUAGE_META[currentVoiceLanguage] || VOICE_LANGUAGE_META.english).code;
}

function getVoiceText(keyOrText) {
    const langMap = VOICE_TEXT[currentVoiceLanguage] || VOICE_TEXT.english;
    return langMap[keyOrText] || keyOrText || '';
}

async function syncVoiceLanguage(language) {
    currentVoiceLanguage = language in VOICE_LANGUAGE_META ? language : 'english';
    localStorage.setItem('voiceLanguage', currentVoiceLanguage);

    const selectEl = document.getElementById('voice-language-select');
    if (selectEl && selectEl.value !== currentVoiceLanguage) {
        selectEl.value = currentVoiceLanguage;
    }

    try {
        await fetch(`/api/voice_language?language=${encodeURIComponent(currentVoiceLanguage)}`, {
            method: 'POST'
        });
    } catch (e) {
        console.warn("Failed to sync voice language:", e);
    }
}

function setVoiceLanguage() {
    const selectEl = document.getElementById('voice-language-select');
    if (!selectEl) return;
    syncVoiceLanguage(selectEl.value);
}

function speakBrowserVoice(text) {
    if (!text || isAudioMuted) return;
    const spokenText = getVoiceText(text);
    if ('speechSynthesis' in window) {
        try {
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance(spokenText);
            msg.rate = 1.0;
            msg.pitch = 1.0;
            msg.volume = 1.0;
            msg.lang = getCurrentVoiceLanguageCode();

            // Pick English voice if available
            if (availableVoices.length > 0) {
                const targetPrefix = msg.lang.split('-')[0];
                const langVoice = availableVoices.find(v => v.lang.toLowerCase().startsWith(targetPrefix));
                const enVoice = availableVoices.find(v => v.lang.toLowerCase().startsWith('en')) || availableVoices[0];
                if (langVoice) {
                    msg.voice = langVoice;
                } else if (enVoice) {
                    msg.voice = enVoice;
                }
            }

            window.speechSynthesis.speak(msg);
        } catch (e) {
            console.warn("Browser SpeechSynthesis failed:", e);
        }
    }
}

async function speakVoicePreview(text) {
    try {
        const response = await fetch('/api/voice_preview', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text,
                language: currentVoiceLanguage
            })
        });

        const contentType = response.headers.get('content-type') || '';
        if (response.ok && contentType.startsWith('audio/')) {
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            const audio = new Audio(audioUrl);
            audio.onended = () => URL.revokeObjectURL(audioUrl);
            await audio.play();
            return;
        }

        const payload = await response.json().catch(() => null);
        if (payload && payload.text) {
            speakBrowserVoice(payload.text);
            return;
        }
    } catch (e) {
        console.warn("Voice preview failed, falling back to browser speech:", e);
    }
    speakBrowserVoice(text);
}

async function pollTelemetry() {
    try {
        const response = await fetch('/api/telemetry');
        if (!response.ok) return;
        const data = await response.json();

        // Reset transient audio suppression once the driver is back to a safe state.
        if (data.state === 'ALERT' || data.state === 'RECOVERING') {
            lastVoiceEvent = null;
            isDhabaDrawerAutoOpened = false;
            lastDhabaVoicePrompt = '';
        }

        // Check for new Voice Alerts from telemetry
        if (data.voice_event && data.voice_event !== lastVoiceEvent) {
            lastVoiceEvent = data.voice_event;
            speakBrowserVoice(data.voice_event);
        }
        if (!data.voice_event && data.state === 'ALERT') {
            lastVoiceEvent = null;
        }

        // 1. Update Metrics
        document.getElementById('val-ear').innerText = data.ear.toFixed(3);
        const marEl = document.getElementById('val-mar');
        marEl.innerText = data.mar.toFixed(3);

        // Highlight MAR if Yawning
        const isYawning = data.is_yawn || data.mar > 0.52 || data.state === 'YAWNING';
        if (isYawning) {
            marEl.className = 'text-xl font-bold font-mono text-amber-400 mt-1 animate-pulse';
        } else {
            marEl.className = 'text-xl font-bold font-mono text-cyan-400 mt-1';
        }

        document.getElementById('val-perclos').innerText = `${data.perclos.toFixed(1)}%`;
        document.getElementById('val-eye-close').innerText = `${data.eye_close_duration.toFixed(2)}s`;
        document.getElementById('fps-counter').innerText = `FPS: ${data.fps.toFixed(1)}`;
        document.getElementById('val-interv').innerText = `L${data.intervention_level}`;
        document.getElementById('val-resp').innerText = data.response_status;

        // 2. Update DVI Gauge Ring & Score
        const score = Math.min(Math.max(data.dvi_score, 0), 100);
        document.getElementById('dvi-score').innerText = `${score.toFixed(1)}%`;

        const circle = document.getElementById('gauge-circle');
        const offset = 440 - (440 * score) / 100;
        circle.style.strokeDashoffset = offset;

        // Color transition
        let color = '#10b981'; // green
        let lvlText = 'LOW RISK';
        if (score >= 75) {
            color = '#ef4444'; // red
            lvlText = 'CRITICAL RISK';
        } else if (score >= 50) {
            color = '#f97316'; // orange
            lvlText = 'HIGH RISK';
        } else if (score >= 25) {
            color = '#eab308'; // yellow
            lvlText = 'MODERATE';
        }

        circle.style.stroke = color;
        const dviLvlEl = document.getElementById('dvi-level');
        dviLvlEl.innerText = lvlText;
        dviLvlEl.style.color = color;

        // 3. Update State Badge & Background styling
        const stateCard = document.getElementById('state-card');
        const statePill = document.getElementById('state-pill');
        const stateText = document.getElementById('state-text');
        const stateSubtext = document.getElementById('state-subtext');
        const criticalOverlay = document.getElementById('critical-overlay');

        stateText.innerText = `STATE: ${data.state}`;

        if (data.state === 'ALERT') {
            statePill.innerText = 'ALERT';
            statePill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30';
            stateCard.className = 'glass-panel p-5 rounded-2xl border border-emerald-500/30 bg-emerald-950/10';
            stateText.className = 'text-2xl font-black tracking-tight text-emerald-400 mt-2';
            stateSubtext.innerText = 'Driver showing normal alertness & eye stability.';
            criticalOverlay.classList.add('hidden');
            dismissEmergencyAlarm(false);
        } else if (data.state === 'YAWNING') {
            statePill.innerText = '🥱 YAWNING DETECTED';
            statePill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-400 border border-amber-500/30 animate-pulse';
            stateCard.className = 'glass-panel p-5 rounded-2xl border border-amber-500/40 bg-amber-950/15';
            stateText.className = 'text-2xl font-black tracking-tight text-amber-400 mt-2';
            stateSubtext.innerText = 'Frequent Yawning detected. Early signs of driver fatigue!';
            criticalOverlay.classList.add('hidden');
            dismissEmergencyAlarm(false);
        } else if (data.state === 'SUSPECTED_DROWSY') {
            statePill.innerText = 'SUSPECTED';
            statePill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-yellow-500/20 text-yellow-400 border border-yellow-500/30';
            stateCard.className = 'glass-panel p-5 rounded-2xl border border-yellow-500/30 bg-yellow-950/10';
            stateText.className = 'text-2xl font-black tracking-tight text-yellow-400 mt-2';
            stateSubtext.innerText = 'Elevated eye closure detected. Confirming temporal persistence...';
            criticalOverlay.classList.add('hidden');
            dismissEmergencyAlarm(false);
        } else if (data.state === 'CONFIRMED_DROWSY' || data.state === 'PERSISTENT_DROWSY' || data.dvi_score >= 75 || data.alarm_triggered) {
            statePill.innerText = 'DROWSY WARNING';
            statePill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-red-500/20 text-red-400 border border-red-500/30';
            stateCard.className = 'glass-panel p-5 rounded-2xl border border-red-500/50 bg-red-950/20';
            stateText.className = 'text-2xl font-black tracking-tight text-red-500 mt-2';
            stateSubtext.innerText = 'CRITICAL: Driver drowsiness confirmed! Loud Audio & Strobe Alert Active!';
            criticalOverlay.classList.remove('hidden');

            // Trigger Loud Emergency Audio Siren & Full-Screen Red Flashing Strobe Alert
            triggerEmergencyStrobeAlert();

            // Auto Open Dhaba Drawer when Drowsy
            if (!isDhabaDrawerAutoOpened) {
                isDhabaDrawerAutoOpened = true;
                openDhabaDrawer();
            }
        } else if (data.state === 'RECOVERING') {
            statePill.innerText = 'RECOVERING';
            statePill.className = 'px-3 py-1 rounded-full text-xs font-bold bg-cyan-500/20 text-cyan-400 border border-cyan-500/30';
            stateCard.className = 'glass-panel p-5 rounded-2xl border border-cyan-500/30 bg-cyan-950/10';
            stateText.className = 'text-2xl font-black tracking-tight text-cyan-400 mt-2';
            stateSubtext.innerText = 'Alertness recovering. Maintaining stability checks...';
            criticalOverlay.classList.add('hidden');
            dismissEmergencyAlarm(false);
        }

        // 4. Update EAR Chart
        if (earChart) {
            earHistory.shift();
            earHistory.push(data.ear);
            earChart.update();
        }

    } catch (e) {
        console.warn("Telemetry fetch failed:", e);
    }
}

// Leaflet.js Dark Mode Interactive Map Controller
let leafletMap = null;
let mapMarkers = [];

function initLeafletMap(lat = 13.1147, lon = 77.5956) {
    const mapContainer = document.getElementById('dhaba-map');
    if (!mapContainer || leafletMap) return;

    leafletMap = L.map('dhaba-map', {
        zoomControl: false
    }).setView([lat, lon], 13);

    // High-contrast CartoDB Dark Matter tiles
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; OpenStreetMap &copy; CARTO',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(leafletMap);

    L.control.zoom({ position: 'topright' }).addTo(leafletMap);
}

function updateMapMarkers(driverLat, driverLon, places) {
    if (!leafletMap) {
        initLeafletMap(driverLat, driverLon);
    }
    if (!leafletMap) return;

    // Clear old markers
    mapMarkers.forEach(m => leafletMap.removeLayer(m));
    mapMarkers = [];

    const bounds = L.latLngBounds();

    // 1. Vehicle Marker (Emerald Glow)
    const vehicleIcon = L.divIcon({
        className: 'custom-vehicle-pin',
        html: `<div style="background:#10b981; width:16px; height:16px; border-radius:50%; border:3px solid #ffffff; box-shadow:0 0 15px #10b981;"></div>`,
        iconSize: [16, 16],
        iconAnchor: [8, 8]
    });

    const driverMarker = L.marker([driverLat, driverLon], { icon: vehicleIcon })
        .addTo(leafletMap)
        .bindPopup(`<div style="font-size:12px; font-weight:bold; color:#10b981;">🚗 Active Vehicle Location</div><div style="font-size:10px; color:#94a3b8;">GPS Telemetry Lock</div>`);

    mapMarkers.push(driverMarker);
    bounds.extend([driverLat, driverLon]);

    // 2. Rest Stop Markers (Amber Glow)
    places.forEach((p, idx) => {
        const isFuel = p.category.includes('Fuel');
        const pinColor = isFuel ? '#3b82f6' : '#f59e0b';
        const pinIconClass = isFuel ? 'fa-gas-pump' : 'fa-utensils';

        const restIcon = L.divIcon({
            className: `custom-rest-pin-${idx}`,
            html: `<div style="background:${pinColor}; width:24px; height:24px; border-radius:50%; border:2px solid #ffffff; box-shadow:0 0 12px ${pinColor}; display:flex; align-items:center; justify-content:center; color:#0f172a; font-size:11px;"><i class="fa-solid ${pinIconClass}"></i></div>`,
            iconSize: [24, 24],
            iconAnchor: [12, 12]
        });

        const popupContent = `
            <div style="padding:4px;">
                <div style="font-size:10px; font-weight:bold; color:${pinColor};">${p.category}</div>
                <div style="font-size:13px; font-weight:bold; color:#ffffff; margin-top:2px;">${p.name}</div>
                <div style="font-size:11px; color:#94a3b8; margin-top:2px;">Distance: ${p.distance_km} km</div>
                <a href="${p.maps_url}" target="_blank" style="display:block; text-align:center; margin-top:8px; padding:5px 10px; background:#f59e0b; color:#0f172a; font-weight:bold; font-size:11px; border-radius:6px; text-decoration:none;">
                    📍 Navigate via Maps
                </a>
            </div>
        `;

        const marker = L.marker([p.lat, p.lon], { icon: restIcon })
            .addTo(leafletMap)
            .bindPopup(popupContent);

        mapMarkers.push(marker);
        bounds.extend([p.lat, p.lon]);
    });

    if (places.length > 0) {
        leafletMap.fitBounds(bounds, { padding: [25, 25] });
    }
}

// Smart Dhaba Assistant Drawer & Fetcher
async function fetchNearbyDhabas() {
    try {
        navigator.geolocation.getCurrentPosition(
            async (pos) => { loadDhabaData(pos.coords.latitude, pos.coords.longitude); },
            () => { loadDhabaData(13.1147, 77.5956); }
        );
    } catch (e) {
        loadDhabaData(13.1147, 77.5956);
    }
}

async function loadDhabaData(lat, lon) {
    const listEl = document.getElementById('dhaba-list');
    const summaryEl = document.getElementById('dhaba-summary');
    try {
        const res = await fetch(`/api/nearby_dhabas?lat=${lat}&lon=${lon}`);
        const data = await res.json();

        if (summaryEl) {
            summaryEl.innerText = data.summary || 'No recommendation summary available right now.';
        }

        if (data.places && data.places.length > 0) {
            listEl.innerHTML = data.places.map((p, i) => `
                <div onclick="focusMapMarker(${i})" class="p-3.5 rounded-xl bg-slate-900 border border-slate-800 hover:border-amber-500/50 transition cursor-pointer flex flex-col gap-2 group">
                    <div class="flex items-start justify-between">
                        <div>
                            <div class="text-xs font-semibold text-amber-400">${p.category}</div>
                            <div class="text-sm font-bold text-white mt-0.5 group-hover:text-amber-300 transition">${p.name}</div>
                        </div>
                        <div class="flex flex-col items-end gap-1">
                            <span class="text-xs px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono font-semibold">${p.distance_km} km</span>
                            <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-500/15 text-emerald-300 border border-emerald-500/20 font-semibold">Score ${Number(p.score || 0).toFixed(1)}</span>
                        </div>
                    </div>
                    <div class="text-[11px] text-slate-400 leading-5">
                        ${(p.reasons || []).slice(0, 3).join(' • ')}
                    </div>
                    <div class="flex items-center gap-2 mt-1">
                        <a href="${p.maps_url}" target="_blank" onclick="event.stopPropagation();" class="flex-1 py-1.5 bg-amber-500/20 hover:bg-amber-500/30 text-amber-300 border border-amber-500/30 rounded-lg text-xs font-semibold text-center transition flex items-center justify-center gap-1.5">
                            <i class="fa-solid fa-diamond-turn-right"></i> Navigate Here
                        </a>
                    </div>
                </div>
            `).join('');

            // Render interactive Leaflet Map markers
            updateMapMarkers(lat, lon, data.places);
            if (data.voice_prompt && data.voice_prompt !== lastDhabaVoicePrompt && isDhabaDrawerAutoOpened) {
                lastDhabaVoicePrompt = data.voice_prompt;
                speakBrowserVoice(data.voice_prompt);
            }
        } else {
            listEl.innerHTML = `<div class="text-xs text-slate-500 text-center py-6">No nearby rest stops found.</div>`;
        }
    } catch (e) {
        listEl.innerHTML = `<div class="text-xs text-slate-500 text-center py-6">Error loading Dhaba list.</div>`;
        if (summaryEl) {
            summaryEl.innerText = 'Error loading the recommendation summary.';
        }
    }
}

function focusMapMarker(index) {
    if (mapMarkers && mapMarkers[index + 1]) {
        const marker = mapMarkers[index + 1];
        leafletMap.setView(marker.getLatLng(), 15, { animate: true });
        marker.openPopup();
    }
}

function openDhabaDrawer() {
    const drawer = document.getElementById('dhaba-drawer');
    drawer.classList.remove('translate-x-full');
    isDhabaDrawerAutoOpened = true;
    fetchNearbyDhabas();
    setTimeout(() => {
        if (leafletMap) leafletMap.invalidateSize();
    }, 350);
}

function toggleDhabaDrawer() {
    const drawer = document.getElementById('dhaba-drawer');
    if (drawer.classList.contains('translate-x-full')) {
        drawer.classList.remove('translate-x-full');
        isDhabaDrawerAutoOpened = true;
        fetchNearbyDhabas();
        setTimeout(() => {
            if (leafletMap) leafletMap.invalidateSize();
        }, 350);
    } else {
        drawer.classList.add('translate-x-full');
    }
}

// 15-Minute Power Nap Countdown Timer
let napInterval = null;
let napSeconds = 15 * 60;

function toggleNapTimer() {
    const btn = document.getElementById('nap-btn');
    if (napInterval) {
        clearInterval(napInterval);
        napInterval = null;
        btn.innerHTML = `<i class="fa-solid fa-play"></i> START NAP`;
        btn.className = "px-4 py-2 bg-amber-500 hover:bg-amber-600 text-slate-950 font-bold text-xs rounded-lg transition flex items-center gap-2";
    } else {
        fetch('/api/nap_alarm', { method: 'POST' });
        btn.innerHTML = `<i class="fa-solid fa-pause"></i> PAUSE`;
        btn.className = "px-4 py-2 bg-red-500 hover:bg-red-600 text-white font-bold text-xs rounded-lg transition flex items-center gap-2";
        napInterval = setInterval(() => {
            if (napSeconds > 0) {
                napSeconds--;
                const m = Math.floor(napSeconds / 60).toString().padStart(2, '0');
                const s = (napSeconds % 60).toString().padStart(2, '0');
                document.getElementById('nap-timer').innerText = `${m}:${s}`;
            } else {
                clearInterval(napInterval);
                alert("15-Minute Power Nap Complete! Time to wake up and drive safely.");
            }
        }, 1000);
    }
}

async function toggleOverlaySettings() {
    const mesh = document.getElementById('toggle-mesh-chk').checked;
    const id = document.getElementById('toggle-id-chk').checked;
    try {
        await fetch(`/api/toggle_settings?mesh=${mesh}&id=${id}`, { method: 'POST' });
    } catch (e) {
        console.warn("Failed to update overlay settings:", e);
    }
}

function downloadSessionSummary(format) {
    window.location.href = `/api/session_summary/download?format=${encodeURIComponent(format)}`;
}

// Emergency Alarm Audio Synth & Strobe Controller
let audioCtx = null;
let osc1 = null;
let osc2 = null;
let lfoOsc = null;
let gainNode = null;
let isAudioMuted = false;
let isAlarmPlaying = false;
let isAlarmDismissed = false;

function initAudioContext() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
}

function startEmergencySiren() {
    if (isAudioMuted || isAlarmPlaying || isAlarmDismissed) return;
    try {
        initAudioContext();
        if (audioCtx.state === 'suspended') {
            audioCtx.resume();
        }

        osc1 = audioCtx.createOscillator();
        osc2 = audioCtx.createOscillator();
        lfoOsc = audioCtx.createOscillator();
        const lfoGain = audioCtx.createGain();
        gainNode = audioCtx.createGain();

        osc1.type = 'sawtooth';
        osc2.type = 'square';

        osc1.frequency.setValueAtTime(880, audioCtx.currentTime);
        osc2.frequency.setValueAtTime(1174.66, audioCtx.currentTime);

        lfoOsc.frequency.setValueAtTime(4, audioCtx.currentTime);
        lfoGain.gain.setValueAtTime(300, audioCtx.currentTime);

        lfoOsc.connect(osc1.frequency);
        lfoOsc.connect(osc2.frequency);

        gainNode.gain.setValueAtTime(0.6, audioCtx.currentTime);

        osc1.connect(gainNode);
        osc2.connect(gainNode);
        gainNode.connect(audioCtx.destination);

        osc1.start();
        osc2.start();
        lfoOsc.start();
        isAlarmPlaying = true;
    } catch (e) {
        console.warn("Audio alarm failed to start:", e);
    }
}

function stopEmergencySiren() {
    if (isAlarmPlaying) {
        try {
            if (osc1) { osc1.stop(); osc1.disconnect(); }
            if (osc2) { osc2.stop(); osc2.disconnect(); }
            if (lfoOsc) { lfoOsc.stop(); lfoOsc.disconnect(); }
            isAlarmPlaying = false;
        } catch (e) { }
    }
}

function toggleAudioAlarm() {
    isAudioMuted = !isAudioMuted;
    const btnText = document.getElementById('audio-status-text');
    const icon = document.getElementById('audio-icon');
    const btn = document.getElementById('toggle-audio-btn');

    if (isAudioMuted) {
        stopEmergencySiren();
        btnText.innerText = "Alarm Audio: OFF";
        icon.className = "fa-solid fa-volume-xmark";
        btn.className = "px-3.5 py-1.5 rounded-lg bg-slate-800 text-slate-400 border border-slate-700 text-xs font-semibold transition flex items-center gap-2";
    } else {
        btnText.innerText = "Alarm Audio: ON";
        icon.className = "fa-solid fa-volume-high";
        btn.className = "px-3.5 py-1.5 rounded-lg bg-rose-500/20 text-rose-300 border border-rose-500/30 text-xs font-semibold hover:bg-rose-500/30 transition flex items-center gap-2";
    }
}

function triggerEmergencyStrobeAlert() {
    document.body.classList.add('fullscreen-strobe-active');
    document.getElementById('emergency-strobe-overlay').classList.remove('hidden');
    startEmergencySiren();
}

function dismissEmergencyAlarm(resetDismissedState = true) {
    isAlarmDismissed = resetDismissedState;
    document.body.classList.remove('fullscreen-strobe-active');
    document.getElementById('emergency-strobe-overlay').classList.add('hidden');
    stopEmergencySiren();
}

function showAudioLogToast(text, type = "info") {
    console.log(`🔊 [AUDIO TOAST] ${text}`);
    let container = document.getElementById('audio-log-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'audio-log-container';
        container.className = 'fixed bottom-5 right-5 z-50 flex flex-col gap-2 pointer-events-none max-w-sm';
        document.body.appendChild(container);
    }
    const toast = document.createElement('div');
    toast.className = `px-4 py-2.5 rounded-xl shadow-2xl backdrop-blur-md text-xs font-mono font-bold flex items-center gap-2 transition-all duration-300 border ${type === 'error' ? 'bg-red-950/90 text-red-300 border-red-500/50' :
        type === 'warn' ? 'bg-amber-950/90 text-amber-300 border-amber-500/50' :
            'bg-slate-900/95 text-emerald-400 border-emerald-500/50'
        }`;
    toast.innerHTML = `<i class="fa-solid ${type === 'error' ? 'fa-triangle-exclamation' : 'fa-volume-high'} animate-bounce"></i> ${text}`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}

function unlockAudioContext() {
    console.log("🔊 [AUDIO UNLOCK] User interaction detected. Unlocking AudioContext & Speech...");
    if (!audioCtx) {
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            console.log("🔊 [AUDIO UNLOCK] Created primary AudioContext. State:", audioCtx.state);
        } catch (e) {
            console.error("❌ [AUDIO UNLOCK ERROR] AudioContext creation failed:", e);
        }
    }
    if (audioCtx && audioCtx.state === 'suspended') {
        audioCtx.resume().then(() => {
            console.log("🔊 [AUDIO UNLOCK] Primary AudioContext resumed! State:", audioCtx.state);
        });
    }
    if ('speechSynthesis' in window && window.speechSynthesis.paused) {
        window.speechSynthesis.resume();
        console.log("🔊 [AUDIO UNLOCK] SpeechSynthesis resumed.");
    }
}

function playDirectBeep(freq = 880, durationMs = 1200) {
    console.log(`🔊 [AUDIO SIREN BEEP] Initializing synth... Freq: ${freq}Hz, Duration: ${durationMs}ms`);
    showAudioLogToast(`Siren Beep Triggered: ${freq}Hz`, 'info');
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) {
            console.error("❌ [AUDIO ERROR] AudioContext is not supported!");
            showAudioLogToast("Error: Web Audio API not supported", 'error');
            return;
        }
        const ctx = new AudioCtx();
        console.log(`🔊 [AUDIO SIREN BEEP] Context created. State: ${ctx.state}`);
        if (ctx.state === 'suspended') {
            ctx.resume().then(() => console.log("🔊 [AUDIO SIREN BEEP] Context resumed successfully!"));
        }
        const osc = ctx.createOscillator();
        const g = ctx.createGain();
        osc.type = 'sawtooth';
        osc.frequency.setValueAtTime(freq, ctx.currentTime);
        g.gain.setValueAtTime(0.8, ctx.currentTime);
        osc.connect(g);
        g.connect(ctx.destination);
        osc.start();
        console.log("🔊 [AUDIO SIREN BEEP] Sound output STARTED!");
        setTimeout(() => {
            try {
                osc.stop();
                ctx.close();
                console.log("🔊 [AUDIO SIREN BEEP] Sound output stopped cleanly.");
            } catch (e) { }
        }, durationMs);
    } catch (e) {
        console.error("❌ [AUDIO ERROR] Direct beep failed:", e);
        showAudioLogToast(`Audio Error: ${e.message}`, 'error');
    }
}

function testAudioSiren() {
    console.log("🔔 [TEST BUTTON CLICKED] Running full Audio Alarm & Speech Diagnostic...");
    showAudioLogToast("Testing Audio Siren & Voice Speech...", "info");
    unlockAudioContext();
    isAlarmDismissed = false;

    // 1. Play immediate direct emergency beep
    playDirectBeep(950, 1500);

    // 3. Spoken voice alert
    speakVoicePreview(VOICE_TEST_PROMPTS[currentVoiceLanguage] || VOICE_TEST_PROMPTS.english);

    // Auto stop test after 3.5 seconds
    setTimeout(() => {
        stopEmergencySiren();
        console.log("🔔 [TEST BUTTON] Alarm test completed.");
    }, 3500);
}

// Global user interaction listener to unlock browser audio policy
document.addEventListener('click', unlockAudioContext);
document.addEventListener('keydown', unlockAudioContext);

// Initialization on DOM Load
document.addEventListener('DOMContentLoaded', () => {
    const languageSelect = document.getElementById('voice-language-select');
    if (languageSelect) {
        languageSelect.value = currentVoiceLanguage;
    }
    syncVoiceLanguage(currentVoiceLanguage);
    initEarChart();
    setInterval(pollTelemetry, 150);
});
