# PayChat — Money Detection API

**Integration Guide for Backend & Mobile Teams**
**Repo:** https://github.com/Akash-Cheerla/paychat-

---

## Overview

A fine-tuned DistilBERT model that detects money-related messages in real-time chat. When a user sends something like "you owe me $25" or "let's split dinner", the API returns structured detection data so the mobile app can trigger a Venmo payment flow.

| Metric | Value |
|--------|-------|
| Architecture | DistilBERT (distilbert-base-uncased) |
| Training data | 5,400 examples (2,700 money, 2,700 non-money) |
| Test accuracy | 100% |
| Test F1 | 100% |
| Inference time | ~300-400ms (CPU), ~20-50ms (GPU) |
| Confidence threshold | 0.65 (configurable) |

---

## Integration Flow

```
User sends message in app
        |
        v
Backend receives message (existing chat flow)
        |
        v
Backend calls:  POST <MODEL_API>/detect
                { "text": "...", "chat_id": "...", "sender": "..." }
        |
        v
Model API returns detection result
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
  "venmo_detection": {
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
  -> Show Venmo popup based on direction field (see below)
```

---

## API Reference

### `POST /detect` — Main Detection Endpoint

Call this for every chat message. Only endpoint the backend needs.

**Request:**
```json
{
  "text": "you owe me $25",
  "chat_id": "room_abc123",
  "message_id": "msg_456",
  "sender": "akash"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | **Yes** | The chat message to analyze |
| `chat_id` | string | No | Chat/room ID for tracking |
| `message_id` | string | No | Message ID (echoed back) |
| `sender` | string | No | Sender name/ID (echoed back) |

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

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `is_money` | boolean | `true` / `false` | Whether the message is money-related |
| `confidence` | float | 0.0 - 1.0 | Model confidence score |
| `trigger_type` | string | `owing_debt`, `bill_splitting`, `direct_amount`, `payment_app`, `general_money` | What category of money mention |
| `direction` | string | `request`, `offer`, `split` | **Who should see the Venmo popup** |
| `detected_amount` | string / null | e.g. `"$25"` | Extracted dollar amount, if present |
| `latency_ms` | float | | Inference time in ms |

**Error responses:**
- `400` — empty text
- `503` — model not loaded yet (server still starting)

---

### Direction Field — Who Gets the Venmo Popup

This is the key field for mobile. It tells you who should see the payment prompt.

| Direction | Meaning | Show popup to | Example messages |
|-----------|---------|---------------|------------------|
| `request` | Sender is asking for money | **Everyone except sender** (they owe the sender) | "you owe me $25", "pay me back", "venmo me" |
| `offer` | Sender is offering to pay | **Sender only** (they want to pay) | "I'll send you $20", "do I owe you?", "let me pay you back", "shall I send the remaining?" |
| `split` | Mutual split | **Everyone in the chat** | "let's split dinner", "halves?", "chip in" |

**Mobile implementation:**
```swift
// iOS / Android pseudocode
if venmo_detection.is_money {
    switch venmo_detection.direction {
    case "request":
        // Sender is requesting money -> show popup to recipients
        if currentUser.id != message.sender_id {
            showVenmoPopup(amount: venmo_detection.detected_amount)
        }
    case "offer":
        // Sender is offering to pay -> show popup to sender
        if currentUser.id == message.sender_id {
            showVenmoPopup(amount: venmo_detection.detected_amount)
        }
    case "split":
        // Split -> show popup to everyone
        showVenmoPopup(amount: venmo_detection.detected_amount)
    }
}
```

---

### `GET /health` — Health Check

Use for monitoring and startup readiness checks.

```json
{
  "status": "ok",
  "device": "cpu",
  "version": {
    "trained_at": "2026-03-22T13:36:00",
    "test_accuracy": 1.0,
    "test_f1": 1.0
  },
  "threshold": 0.65,
  "uptime_reqs": 0
}
```

### `GET /metrics` — Inference Stats

```json
{
  "requests": 142,
  "money_detected": 37,
  "detection_rate": 0.2606,
  "avg_latency_ms": 312.45,
  "started_at": "2026-03-24T08:17:03"
}
```

### `POST /reload` — Hot-Reload Model

Reload model weights from disk without restarting the server. Use after retraining.

```json
{ "status": "ok", "loaded_at": "2026-03-24T10:00:00" }
```

### `WebSocket /ws/detect` — Real-Time Detection (Alternative)

If the backend uses WebSocket instead of REST for chat, connect to this endpoint.

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

---

## Detection Categories

| trigger_type | What it detects | Examples |
|-------------|-----------------|----------|
| `owing_debt` | Someone owes or is owed money | "you owe me $25", "pay me back", "I owe you for lunch" |
| `bill_splitting` | Splitting a bill or expense | "let's split dinner", "halves?", "chip in $10 each" |
| `direct_amount` | A dollar amount with payment context | "that's $50", "it was 30 bucks" |
| `payment_app` | Mention of a payment platform | "venmo me", "send it on cashapp", "zelle me" |
| `general_money` | Other money-related requests | "can you cover me?", "spot me", "lend me some" |

**What it intentionally ignores:**
- General finance talk ("the stock market is up")
- Prices without payment intent ("that shirt is $50")
- News/articles mentioning money
- Song lyrics, quotes with dollar signs

---

## Deployment

### Docker (Recommended)

```bash
git clone https://github.com/Akash-Cheerla/paychat-.git
cd paychat-
docker build -t paychat-api .
docker run -p 8000:8000 paychat-api
```

The API will be available at `http://localhost:8000`.

### Without Docker

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
| `MODEL_DIR` | auto-detected | Absolute path to the `saved_model/` directory |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum confidence to flag as money (0.0 - 1.0) |

### Model Files (included in repo)

```
model/saved_model/
  config.json              # Model architecture config
  model.safetensors        # Trained model weights (255 MB, tracked with Git LFS)
  tokenizer.json           # Tokenizer vocabulary
  tokenizer_config.json    # Tokenizer settings
  training_report.json     # Training metrics and history
```

**Note:** `model.safetensors` is stored with Git LFS. Make sure `git lfs` is installed before cloning:
```bash
git lfs install
git clone https://github.com/Akash-Cheerla/paychat-.git
```

### Production Recommendations

- **GPU:** Deploy on a GPU instance (AWS g4dn.xlarge, GCP T4) for ~20-50ms inference vs ~300-400ms on CPU
- **Workers:** Use `--workers 2` for CPU, `--workers 1` for GPU (model is loaded per-worker)
- **Health check:** Hit `GET /health` to confirm model is loaded before routing traffic
- **Scaling:** The API is stateless — scale horizontally behind a load balancer

---

## API Docs (Swagger)

Once deployed, interactive API documentation is available at:
```
http://<your-server>:8000/docs
```

---

## Dependencies

```
torch>=2.2.0
transformers>=4.39.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pydantic>=2.6.0
```

Full list in `requirements.txt`.

---

## Questions?

Reach out to Akash.
