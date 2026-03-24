# SceneFlow API

**API-first engine that converts raw text into structured, multi-scene explainer videos.**

SceneFlow uses LLMs to build a JSON Scene DSL, renders each scene as HTML via a headless browser, generates TTS narration, and stitches everything together with FFmpeg.

---

## Architecture

```
Client                     SceneFlow API                    Workers
  │                            │                               │
  ├── POST /api/v1/draft ─────►│  LLM (GPT-4o-mini)           │
  │◄──── VideoDraft JSON ──────┤  → Structured Scene DSL       │
  │                            │                               │
  ├── POST /api/v1/render ────►│───── Celery Task ────────────►│
  │◄──── { job_id } ──────────┤                               │  TTS → HTML → Playwright
  │                            │                               │  → FFmpeg → .mp4
  ├── GET /api/v1/jobs/{id} ──►│◄──── Result ─────────────────┤
  │◄──── { status, url } ─────┤                               │
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API Framework | FastAPI (Python 3.11+) |
| LLM | OpenAI GPT-4o-mini (Structured Outputs) |
| TTS | OpenAI TTS API (`tts-1`) |
| Renderer | Playwright (Headless Chromium) |
| Compositor | FFmpeg via `ffmpeg-python` |
| Task Queue | Celery + Redis |

## Project Structure

```
SceneFlow/
├── .env.example           # Environment variable template
├── .gitignore
├── README.md
└── sceneflow/
    ├── __init__.py
    ├── main.py            # FastAPI routes
    ├── schemas.py         # Pydantic Scene DSL models
    ├── config.py          # Settings & env var loading
    ├── llm_engine.py      # OpenAI structured output logic
    ├── celery_worker.py   # Celery app + render task
    ├── video_pipeline.py  # TTS, Playwright, FFmpeg pipeline
    ├── requirements.txt   # Python dependencies
    └── templates/
        └── base_scene.html  # Jinja2 visual template
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Redis (running locally or via Docker)
- FFmpeg (`brew install ffmpeg`)
- An OpenAI API key

### 2. Setup

```bash
# Clone and enter the project
cd SceneFlow

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r sceneflow/requirements.txt

# Install Playwright browsers
playwright install chromium

# Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:alpine

# OR using Homebrew
brew services start redis
```

### 4. Start the API Server

```bash
uvicorn sceneflow.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the Celery Worker

```bash
celery -A sceneflow.celery_worker worker --loglevel=info
```

### 6. Use the API

**Generate a draft:**

```bash
curl -X POST http://localhost:8000/api/v1/draft \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "How REST APIs work",
    "audience": "beginners",
    "brand_colors": ["#4F46E5", "#10B981"]
  }'
```

**Render the video** (paste the returned draft JSON):

```bash
curl -X POST http://localhost:8000/api/v1/render \
  -H "Content-Type: application/json" \
  -d '{ "draft": <PASTE_DRAFT_JSON_HERE> }'
```

**Check job status:**

```bash
curl http://localhost:8000/api/v1/jobs/<JOB_ID>
```

### 7. Interactive Docs

Visit [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/draft` | Generate Scene DSL from a topic |
| `POST` | `/api/v1/render` | Queue video rendering from a draft |
| `GET` | `/api/v1/jobs/{job_id}` | Poll rendering status & get result |

## License

MIT
