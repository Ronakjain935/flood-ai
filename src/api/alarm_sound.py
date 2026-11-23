"""
Flood-AI: Emergency Siren and Audible Alarm System.
Generates cross-browser audio warnings using standard library PCM WAV synthesis
and client-side Web Audio API oscillators for zero-dependency real-time alarming.
"""

import base64
import io
import math
import struct


def generate_emergency_siren_wav(duration_sec: float = 2.5, sample_rate: int = 22050) -> str:
    """
    Synthesizes a 16-bit mono dual-tone pulsing emergency flood siren.
    Returns base64 encoded audio/wav data URI for instant browser playback.
    """
    total_samples = int(duration_sec * sample_rate)
    samples = []
    
    # Dual-tone sweeping siren: 750 Hz to 1250 Hz oscillation at 2 Hz modulation
    for i in range(total_samples):
        t = i / sample_rate
        # Frequency modulation between 750Hz and 1250Hz
        freq = 950.0 + 300.0 * math.sin(2.0 * math.pi * 1.8 * t)
        # Amplitude envelope with slight ramp-in
        amplitude = 0.35 * min(1.0, t * 8.0)
        # Pulse tone (sawtooth-like blend)
        val = int(32767.0 * amplitude * (0.8 * math.sin(2.0 * math.pi * freq * t) + 0.2 * math.sin(4.0 * math.pi * freq * t)))
        # Clamp to 16-bit signed integer limits
        val = max(-32767, min(32767, val))
        samples.append(struct.pack("<h", val))

    raw_pcm = b"".join(samples)
    data_size = len(raw_pcm)

    # Construct standard 44-byte RIFF/WAVE header manually
    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", 36 + data_size))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", 16))            # Subchunk1Size (16 for PCM)
    header.extend(struct.pack("<H", 1))             # AudioFormat (1 = PCM)
    header.extend(struct.pack("<H", 1))             # NumChannels (1 = Mono)
    header.extend(struct.pack("<I", sample_rate))   # SampleRate
    header.extend(struct.pack("<I", sample_rate * 2))  # ByteRate (SampleRate * NumChannels * BitsPerSample/8)
    header.extend(struct.pack("<H", 2))             # BlockAlign (NumChannels * BitsPerSample/8)
    header.extend(struct.pack("<H", 16))            # BitsPerSample (16 bits)
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))

    full_wav = bytes(header) + raw_pcm
    b64_audio = base64.b64encode(full_wav).decode("ascii")
    return f"data:audio/wav;base64,{b64_audio}"


def get_web_audio_alarm_html(is_active: bool, region_name: str, max_probability: float) -> str:
    """
    Renders an interactive HTML5/JS audio siren control with animated pulsing banner
    and Web Audio API oscillator for instant real-time sound playback in Streamlit.
    """
    prob_percent = int(max_probability * 100)
    wav_uri = generate_emergency_siren_wav(duration_sec=2.2)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <style>
        @keyframes emergency-pulse {{
          0% {{ background-color: #991b1b; box-shadow: 0 0 15px rgba(239, 68, 68, 0.6); }}
          50% {{ background-color: #dc2626; box-shadow: 0 0 30px rgba(239, 68, 68, 1.0); }}
          100% {{ background-color: #991b1b; box-shadow: 0 0 15px rgba(239, 68, 68, 0.6); }}
        }}
        @keyframes siren-icon-spin {{
          0% {{ transform: rotate(-15deg) scale(1); }}
          50% {{ transform: rotate(15deg) scale(1.2); }}
          100% {{ transform: rotate(-15deg) scale(1); }}
        }}
        .alarm-banner {{
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 14px 20px;
          border-radius: 10px;
          color: #ffffff;
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          animation: emergency-pulse 1.4s infinite ease-in-out;
          border: 2px solid #f87171;
          margin-bottom: 12px;
        }}
        .alarm-title {{
          display: flex;
          align-items: center;
          gap: 12px;
        }}
        .siren-icon {{
          font-size: 32px;
          display: inline-block;
          animation: siren-icon-spin 0.8s infinite alternate;
        }}
        .alarm-text b {{
          font-size: 16px;
          letter-spacing: 0.5px;
          display: block;
        }}
        .alarm-text span {{
          font-size: 13px;
          opacity: 0.95;
        }}
        .btn-controls {{
          display: flex;
          gap: 10px;
          align-items: center;
        }}
        .alarm-btn {{
          background: #ffffff;
          color: #b91c1c;
          border: none;
          font-weight: 700;
          font-size: 13px;
          padding: 8px 16px;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.2s;
          box-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }}
        .alarm-btn:hover {{
          background: #fee2e2;
          transform: translateY(-1px);
        }}
      </style>
    </head>
    <body style="margin: 0; background: transparent;">
      <div class="alarm-banner">
        <div class="alarm-title">
          <span class="siren-icon">🚨</span>
          <div class="alarm-text">
            <b>CRITICAL FLOOD ALARM SOUNDING: {region_name.upper()}</b>
            <span>High flood probability detected ({prob_percent}% risk). Immediate life safety precautions in effect.</span>
          </div>
        </div>
        <div class="btn-controls">
          <button id="sirenBtn" class="alarm-btn" onclick="toggleWebAudioSiren()">🔊 Sound Siren Now</button>
        </div>
      </div>

      <!-- Embedded Autoplay Audio Element for Immediate Trigger -->
      <audio id="sirenAudio" preload="auto" loop>
        <source src="{wav_uri}" type="audio/wav">
      </audio>

      <script>
        let isAudioPlaying = false;
        let audioElem = document.getElementById("sirenAudio");
        let webAudioCtx = null;
        let oscillator = null;
        let gainNode = null;

        function startSyntheticSiren() {{
          try {{
            const AudioCtx = window.AudioContext || window.webkitAudioContext;
            if (!AudioCtx) return;
            if (!webAudioCtx) {{
              webAudioCtx = new AudioCtx();
            }}
            if (webAudioCtx.state === 'suspended') {{
              webAudioCtx.resume();
            }}
            oscillator = webAudioCtx.createOscillator();
            gainNode = webAudioCtx.createGain();
            oscillator.type = 'sawtooth';

            const now = webAudioCtx.currentTime;
            // 2-tone alarm frequency sweep
            oscillator.frequency.setValueAtTime(800, now);
            for (let i = 0; i < 30; i++) {{
              oscillator.frequency.linearRampToValueAtTime(1200, now + (i * 0.7) + 0.35);
              oscillator.frequency.linearRampToValueAtTime(800, now + (i * 0.7) + 0.7);
            }}

            gainNode.gain.setValueAtTime(0.3, now);
            oscillator.connect(gainNode);
            gainNode.connect(webAudioCtx.destination);
            oscillator.start();
            isAudioPlaying = true;
            document.getElementById("sirenBtn").innerText = "🔇 Silence Siren";
            document.getElementById("sirenBtn").style.background = "#fee2e2";
          }} catch (err) {{
            console.warn("WebAudio synth fallback to HTML5 audio:", err);
            if (audioElem) {{
              audioElem.play().catch(e => console.log("Autoplay policy restricted:", e));
              isAudioPlaying = true;
              document.getElementById("sirenBtn").innerText = "🔇 Silence Siren";
            }}
          }}
        }}

        function stopSyntheticSiren() {{
          if (oscillator) {{
            try {{ oscillator.stop(); oscillator.disconnect(); }} catch(e) {{}}
            oscillator = null;
          }}
          if (audioElem) {{
            try {{ audioElem.pause(); audioElem.currentTime = 0; }} catch(e) {{}}
          }}
          isAudioPlaying = false;
          document.getElementById("sirenBtn").innerText = "🔊 Sound Siren";
          document.getElementById("sirenBtn").style.background = "#ffffff";
        }}

        function toggleWebAudioSiren() {{
          if (isAudioPlaying) {{
            stopSyntheticSiren();
          }} else {{
            startSyntheticSiren();
          }}
        }}

        // Attempt automated sound playback if active
        {'startSyntheticSiren();' if is_active else ''}
      </script>
    </body>
    </html>
    """
