# PayChat Money Detection API — Integration Guide

**For: Samyak (Backend) + App Dev Team**
**From: Akash**
**Date: April 3, 2026**

---

## What This Is

A fine-tuned DistilBERT model that detects money-related messages in real-time chat. When a user sends something like "you owe me $25" or "let's split dinner", the API detects it and returns structured data so the mobile app can show a Venmo popup.

**Live demo:** https://fyoe.onrender.com
**GitHub repo:** https://github.com/Akash-Cheerla/paychat-

---

## Model Stats

| Metric | Value |
|--------|-------|
| Architecture | DistilBERT (distilbert-base-uncased) |
| Training data | 5,400 examples (2,700 money, 2,700 non-money) |
| Test accuracy | 100% |
| Test F1 | 100% |
| Test AUC | 1.0 |
| Inference time | ~300-400ms (CPU), ~20-50ms (GPU) |
| Confidence threshold | 0.65 |

---

## API Endpoints

### 1. `POST /detect` — Main Detection Endpoint

This is the primary endpoint the backend should call for every chat message.

**Request:**
```json
POST /detect
Content-Type: application/json

{
  "text": "you owe me $25",
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | The chat message to analyze |
| `chat_id` | string | No | Chat/room ID for multi-user tracking |
| `message_id` | string | No | Unique message ID (pass-through) |
| `sender` | string | No | Sender's name/ID (pass-through) |

**Response:**
```json
{
  "is_money": true,
  "confidence": 0.9979,
  "trigger_type": "owing_debt",
  "direction": "request",
  "detected_amount": "$25",
  "latency_ms": 342.15,
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `is_money` | boolean | Whether the message is money-related |
| `confidence` | float | 0.0 - 1.0, model confidence |
| `trigger_type` | string | Category: `owing_debt`, `bill_splitting`, `direct_amount`, `payment_app`, `general_money` |
| `direction` | string | Who should see the Venmo popup: `request`, `offer`, `split` (see below) |
| `detected_amount` | string or null | Extracted dollar amount (e.g. "$25") |
| `latency_ms` | float | Model inference time in ms |
| `chat_id` | string or null | Echoed back from request |
| `message_id` | string or null | Echoed back from request |
| `sender` | string or null | Echoed back from request |

### Direction Field — Who Gets the Venmo Popup

| Direction | Meaning | Popup shows for | Example messages |
|-----------|---------|-----------------|------------------|
| `request` | Sender is asking for money | **Recipients** (they owe) | "you owe me $25", "pay me back", "venmo me" |
| `offer` | Sender is offering to pay | **Sender** (they owe) | "I'll send you $20", "do I owe you?", "let me pay you back" |
| `split` | Mutual split | **Everyone** | "let's split dinner", "halves?", "chip in" |

**Mobile app logic:**
```
if direction == "request":
    show popup to everyone EXCEPT the sender
elif direction == "offer":
    show popup to the sender only
elif direction == "split":
    show popup to everyone in the chat
```

### 2. `GET /health` — Health Check

```json
GET /health

Response:
{
  "status": "ok",
  "device": "cpu",
  "model_dir": "/app/saved_model",
  "loaded_at": "2026-03-24T08:17:03.132143",
  "version": {
    "trained_at": "2026-03-22T13:36:00",
    "test_accuracy": 1.0,
    "test_f1": 1.0
  },
  "threshold": 0.65,
  "uptime_reqs": 0
}
```

### 3. `GET /metrics` — Live Stats

```json
GET /metrics

Response:
{
  "requests": 142,
  "money_detected": 37,
  "detection_rate": 0.2606,
  "avg_latency_ms": 312.45,
  "started_at": "2026-03-24T08:17:03"
}
```

### 4. `POST /reload` — Hot-Reload Model

Reload model from disk without restarting the server. Used after retraining.

```json
POST /reload

Response:
{ "status": "ok", "loaded_at": "2026-03-24T10:00:00" }
```

### 5. `WebSocket /ws/detect` — Real-Time Detection (Optional)

For backend WebSocket integration instead of REST.

```json
// Send:
{ "text": "you owe me $25", "message_id": "123", "sender": "akash" }

// Receive:
{
  "text": "you owe me $25",
  "message_id": "123",
  "sender": "akash",
  "venmo_detection": {
    "is_money": true,
    "confidence": 0.9979,
    "trigger_type": "owing_debt",
    "detected_amount": "$25",
    "latency_ms": 342.15
  }
}
```

### 6. `WebSocket /ws/chat/{room_id}` — Live Chat Room (Demo Only)

This powers the demo UI at the root URL. Not needed for production — the mobile app has its own chat.

---

## Integration Flow

```
User sends message in app
        |
        v
Backend receives message (existing flow)
        |
        v
Backend calls POST /detect  <-- just add this step
  { "text": "...", "chat_id": "...", "sender": "..." }
        |
        v
API returns detection result
        |
        v
Backend attaches venmo_detection to the message payload
        |
        v
Recipient's app receives enriched message:
{
  "id": "msg_456",
  "text": "you owe me $25",
  "sender": "akash",
  "timestamp": "...",
  "venmo_detection": {       <-- new field
    "is_money": true,
    "confidence": 0.9979,
    "trigger_type": "owing_debt",
    "direction": "request",
    "detected_amount": "$25"
  }
}
        |
        v
Mobile app checks: if venmo_detection.is_money == true
  -> Show Venmo popup based on direction field
```

---

## Deployment

### Option A: Use the Render Instance (Already Live)

The API is already deployed at **https://fyoe.onrender.com**

Backend just needs to call:
```
POST https://fyoe.onrender.com/detect
```

**Notes:**
- Render free tier sleeps after 15 min of inactivity. First request after sleep takes ~30-60s to cold-start. Upgrade to paid tier ($7/mo) to keep it always on.
- On free-tier CPU, model inference is ~300-400ms warm, ~20s on cold start. For production, deploy on a GPU instance (AWS/GCP) for ~20-50ms inference, or use the keyword-based fast detection built into the `/ws/chat` endpoint for instant results.

### Option B: Deploy on Your Own Server

**Docker (recommended):**
```bash
git clone https://github.com/Akash-Cheerla/paychat-.git
cd paychat-
docker build -t paychat-api .
docker run -p 8000:8000 paychat-api
```

**Without Docker:**
```bash
git clone https://github.com/Akash-Cheerla/paychat-.git
cd paychat-
pip install -r requirements.txt
cd api
MODEL_DIR=../model/saved_model uvicorn app:app --host 0.0.0.0 --port 8000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `../model/saved_model` | Path to the trained model directory |
| `DEMO_DIR` | `../demo` | Path to demo HTML (optional) |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum confidence to flag as money |

### Required Files in MODEL_DIR

```
saved_model/
  config.json              # Model architecture config
  model.safetensors        # Model weights (255 MB)
  tokenizer.json           # Tokenizer vocab
  tokenizer_config.json    # Tokenizer settings
  training_report.json     # Training metrics
```

---

## What the Model Detects

### Categories (trigger_type)

| Category | Examples |
|----------|----------|
| `owing_debt` | "you owe me $25", "pay me back", "I owe you for lunch" |
| `bill_splitting` | "let's split dinner", "halves?", "chip in $10 each" |
| `direct_amount` | "that's $50", "it was 30 bucks" |
| `payment_app` | "venmo me", "send it on cashapp", "zelle me the money" |
| `general_money` | "can you cover me?", "I need money", "lend me some" |

### What It Doesn't Detect (on purpose)

- General finance talk: "the stock market is up"
- Prices without context: "that shirt is $50" (no payment intent)
- News/articles about money
- Song lyrics, quotes with dollar signs

---

## Quick Test

```bash
# Money message
curl -X POST https://fyoe.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "you owe me $25", "chat_id": "test", "sender": "akash"}'

# Not money
curl -X POST https://fyoe.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "hey what time is dinner tonight", "chat_id": "test", "sender": "akash"}'

# Offer to pay (direction = "offer")
curl -X POST https://fyoe.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "I owe you for lunch, let me venmo you", "chat_id": "test", "sender": "akash"}'

# Split (direction = "split")
curl -X POST https://fyoe.onrender.com/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "should we split the bill 3 ways?", "chat_id": "test", "sender": "akash"}'
```

---

## Swagger Docs

Full interactive API docs available at:
**https://fyoe.onrender.com/docs**

---

## Questions?

Reach out to Akash. The model, API, training data, and everything is in the GitHub repo.
