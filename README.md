# Jarvis

Local-first voice assistant scaffold: **faster-whisper** for speech-to-text, **Piper** for speech, **Claude** (Anthropic) for reasoning and tool use, and **SQLite** for memory.

**Languages:** Python 3.10–3.13 (assistant), HTML/CSS/JavaScript (HUD).

Logic is not implemented yet — each module is a documented stub.

## Layout

```
jarvis/
  main.py              # listen → think → speak loop
  config.py            # env-driven settings
  audio/
    listener.py        # mic + wake word + faster-whisper
    speaker.py         # piper-tts playback
  brain/
    llm.py             # Anthropic Messages API + tool loop
    tools.py           # functions Claude can call
  memory/
    store.py           # SQLite facts + turn log
  requirements.txt
  .env.example
```

## Requirements

- Python **3.10–3.13**
- A microphone and speakers
- An [Anthropic API key](https://console.anthropic.com/) (STT and TTS stay on-device)
- **PortAudio** (needed by `sounddevice`)
  - macOS: `brew install portaudio`
  - Debian/Ubuntu: `sudo apt install libportaudio2`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

Pinned versions in `requirements.txt` are chosen so **faster-whisper** and **piper-tts** share `onnxruntime==1.20.1` (`<2`), which both packages require.

When you implement TTS, download a [Piper voice](https://github.com/OHF-Voice/piper1-gpl) (ONNX + JSON) into `models/` and point `PIPER_VOICE` at it.

## Run

```bash
python main.py
```

This currently raises `NotImplementedError` until the stubs are filled in.

## Stack

| Piece | Package | Runs |
| --- | --- | --- |
| Speech-to-text | `faster-whisper==1.2.1` | Local |
| Wake word / mic | `sounddevice` + Whisper window | Local |
| LLM + tools | `anthropic==1.1.0` | Anthropic API |
| Text-to-speech | `piper-tts==1.7.0` | Local |
| Memory | stdlib `sqlite3` | Local |
