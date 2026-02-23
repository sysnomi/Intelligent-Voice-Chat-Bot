# 🎙️ Voice AI Insight Engine

Real-time voice AI backend — STT → LLM Entity/Sentiment Extraction → TTS — built with FastAPI, LangGraph, and WebSockets.

## Quick Start

### 1. Clone & set up environment
```bash
cd voice-chat
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
copy .env.example .env
# Edit .env and fill in your API keys
```

### 3. Run the server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open http://localhost:8000/docs for the interactive API docs.

---

## Provider Configuration (`.env`)

| Variable | Options | Default |
|---|---|---|
| `STT_PROVIDER` | `azure` \| `groq` | `azure` |
| `LLM_PROVIDER` | `openai` \| `gemini` \| `groq` \| `azure_openai` | `openai` |
| `TTS_PROVIDER` | `elevenlabs` \| `kokoro` | `elevenlabs` |

Switch any provider by changing the variable — no code changes required.

---

## WebSocket Protocol

Connect to: `ws://localhost:8000/ws/voice/new`

**Client → Server:**
| Frame | Type | Description |
|---|---|---|
| Binary | Audio | Raw PCM chunks (16kHz, 16-bit mono) |
| JSON | `{"type": "audio_end"}` | Signal end of utterance → trigger pipeline |
| JSON | `{"type": "interrupt"}` | Stop current TTS playback |

**Server → Client:**
| `type` | Description |
|---|---|
| `session_ready` | Connection confirmed with `session_id` |
| `transcript_partial` | Streaming interim STT text |
| `transcript_final` | Committed final transcript |
| `extraction` | JSON with entities, relationships, sentiment, intent |
| `audio_chunk` | Base64 MP3 chunk for playback |
| `audio_done` | All TTS chunks sent |
| `interrupt` | Acknowledge interruption |

---

## Project Structure

```
voice-chat/
├── app/
│   ├── main.py           # FastAPI entry point
│   ├── config.py         # Provider config (pydantic-settings)
│   ├── api/
│   │   ├── websocket.py  # WS endpoint /ws/voice/{session_id}
│   │   └── rest.py       # REST: /api/health, /api/sessions
│   ├── stt/              # STT: Azure + Groq + factory
│   ├── brain/            # LangGraph: state, schemas, prompts, graph
│   ├── tts/              # TTS: ElevenLabs + Kokoro + factory
│   ├── core/             # Session manager + pipeline orchestrator
│   └── utils/            # Logging + audio helpers
├── tests/
├── .env.example
├── Dockerfile
└── requirements.txt
```

---

## REST API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | App info + active providers |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/sessions` | Create a session |
| `GET` | `/api/sessions/{id}` | Get session state |
| `DELETE` | `/api/sessions/{id}` | End session |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker build -t voice-ai .
docker run -p 8000:8000 --env-file .env voice-ai
```
