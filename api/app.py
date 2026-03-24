"""
PayChat Money Detector — Inference API
FastAPI server with:
  - POST /detect     → single message detection
  - WS   /ws/detect  → real-time WebSocket detection for chat apps
  - GET  /health     → health check (model version, accuracy, uptime)
  - GET  /metrics    → inference stats

Designed for backend to intercept WebSocket messages and enrich them
with venmo_detection field before forwarding to mobile clients.
"""

import asyncio
import json
import os
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──
MODEL_DIR      = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent.parent / "model" / "saved_model")))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MAX_LEN        = 128
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Global state ──
model_state = {
    "model":     None,
    "tokenizer": None,
    "version":   None,
    "loaded_at": None,
}

stats = {
    "requests": 0,
    "money_detected": 0,
    "started_at": datetime.utcnow().isoformat(),
    "avg_latency_ms": 0,
    "_latency_sum": 0,
}


# ── Model loading ──
def load_model(model_dir: Path = MODEL_DIR):
    """Load or reload model from disk. Thread-safe for hot-swap."""
    logger.info(f"Loading model from {model_dir}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model     = DistilBertForSequenceClassification.from_pretrained(str(model_dir))
    model     = model.to(DEVICE)
    model.eval()

    version = None
    report_path = model_dir / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)
        version = {
            "trained_at":     report.get("trained_at"),
            "test_accuracy":  report.get("test_accuracy"),
            "test_f1":        report.get("test_f1"),
        }

    model_state["model"]     = model
    model_state["tokenizer"] = tokenizer
    model_state["version"]   = version
    model_state["loaded_at"] = datetime.utcnow().isoformat()
    logger.info(f"Model loaded. Accuracy: {version['test_accuracy']:.2%}" if version else "Model loaded.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App ──
app = FastAPI(
    title="PayChat Money Detector",
    description="Real-time money detection for chat apps. Detects: owing/debt, bill splitting, direct amounts, Venmo/CashApp mentions.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Inference ──
def run_inference(text: str) -> dict:
    """Run the model on a single text. Returns detection result."""
    t0 = time.time()

    tokenizer = model_state["tokenizer"]
    model     = model_state["model"]

    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

    confidence = float(probs[1])  # P(money)
    is_money   = confidence >= CONFIDENCE_THRESHOLD

    # Extract dollar amount if present
    import re
    amount_match = re.search(r'\$[\d,]+(?:\.\d{1,2})?|\b\d+\s*\$|\b\d+\s*(?:dollars?|bucks?)\b', text, re.IGNORECASE)
    detected_amount = amount_match.group(0) if amount_match else None

    # Classify trigger type and direction
    trigger = _classify_trigger(text, is_money)
    direction = _classify_direction(text) if is_money else None

    latency_ms = (time.time() - t0) * 1000

    # Update stats
    stats["requests"]     += 1
    stats["_latency_sum"] += latency_ms
    stats["avg_latency_ms"] = stats["_latency_sum"] / stats["requests"]
    if is_money:
        stats["money_detected"] += 1

    return {
        "is_money":        is_money,
        "confidence":      round(confidence, 4),
        "trigger_type":    trigger,
        "direction":       direction,
        "detected_amount": detected_amount,
        "latency_ms":      round(latency_ms, 2),
    }


def _classify_trigger(text: str, is_money: bool) -> Optional[str]:
    """Classify what kind of money trigger was detected."""
    if not is_money:
        return None
    t = text.lower()
    if any(w in t for w in ["venmo", "cashapp", "cash app", "zelle", "apple pay"]):
        return "payment_app"
    if any(w in t for w in ["split", "halves", "half", "divide", "share", "chip in"]):
        return "bill_splitting"
    if any(w in t for w in ["owe", "owed", "pay me back", "pay back", "pay you back"]):
        return "owing_debt"
    if any(c in t for c in ["$"]) or any(w in t for w in ["dollars", "bucks"]):
        return "direct_amount"
    return "general_money"


def _classify_direction(text: str) -> str:
    """
    Determine money flow direction from the sender's perspective.
      'request'  = sender is asking for money  -> popup shows for recipients
      'offer'    = sender is offering to pay    -> popup shows for sender
      'split'    = mutual split                 -> popup shows for everyone
    """
    t = text.lower()

    # Sender is OFFERING to pay / acknowledging they owe
    offer_patterns = [
        "i owe", "i'll pay", "i'll send", "let me pay", "let me send",
        "i can pay", "i can send", "i'll venmo", "i'll cashapp", "i'll zelle",
        "shall i send", "should i send", "want me to send", "want me to pay",
        "do i owe", "how much do i owe", "i need to pay", "paying you",
        "send you", "pay you back", "i'll cover", "let me cover",
        "i got you", "my treat", "i'll get this", "on me",
        "sending you", "lemme pay", "lemme send", "ima send", "ima pay",
        "i can venmo", "i can cashapp", "i can zelle",
    ]
    for p in offer_patterns:
        if p in t:
            return "offer"

    # Sender is REQUESTING money
    request_patterns = [
        "you owe", "owe me", "pay me", "send me", "pay up",
        "venmo me", "cashapp me", "zelle me", "where's my",
        "give me", "front me", "spot me", "cover me",
        "you still owe", "pay me back", "need my money",
        "hit me with", "throw me",
    ]
    for p in request_patterns:
        if p in t:
            return "request"

    # Split / mutual
    split_patterns = [
        "split", "halves", "half", "divide", "each",
        "chip in", "go dutch", "share the",
    ]
    for p in split_patterns:
        if p in t:
            return "split"

    # Default: if it mentions a payment app generically, treat as offer
    # (e.g. "venmo me your handle" is request, but "just venmo" is ambiguous)
    return "request"


# ── Request/Response schemas ──
class DetectRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None
    message_id: Optional[str] = None
    sender: Optional[str] = None


class DetectResponse(BaseModel):
    is_money: bool
    confidence: float
    trigger_type: Optional[str]
    direction: Optional[str]
    detected_amount: Optional[str]
    latency_ms: float
    chat_id: Optional[str]
    message_id: Optional[str]
    sender: Optional[str]


# ── Routes ──
@app.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """
    Detect money-related content in a chat message.

    Returns:
      - is_money: true/false
      - confidence: 0.0–1.0 (model probability)
      - trigger_type: what kind of money mention (owing_debt, bill_splitting, etc.)
      - detected_amount: extracted dollar amount if present
      - latency_ms: inference time
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if model_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = run_inference(req.text)
    return DetectResponse(
        **result,
        chat_id=req.chat_id,
        message_id=req.message_id,
        sender=req.sender,
    )


@app.websocket("/ws/detect")
async def ws_detect(websocket: WebSocket):
    """
    WebSocket endpoint for real-time message detection.

    Backend sends:
      { "text": "msg", "message_id": "123", "sender": "alice" }

    Server returns:
      { ...original message..., "venmo_detection": { is_money, confidence, trigger_type, detected_amount } }

    Mobile clients receive enriched payload — no changes needed on their side.
    """
    await websocket.accept()
    logger.info(f"WS client connected: {websocket.client}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            text = msg.get("text", "")
            if not text.strip():
                await websocket.send_text(json.dumps({"error": "text required"}))
                continue

            result = run_inference(text)

            # Return original message enriched with detection
            response = {
                **msg,
                "venmo_detection": {
                    "is_money":        result["is_money"],
                    "confidence":      result["confidence"],
                    "trigger_type":    result["trigger_type"],
                    "detected_amount": result["detected_amount"],
                    "latency_ms":      result["latency_ms"],
                }
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        logger.info(f"WS client disconnected: {websocket.client}")


# ── Live Chat Room ──
# Multi-user WebSocket chat with real-time money detection
chat_rooms: dict[str, set[WebSocket]] = {}   # room_id -> set of connected sockets
chat_users: dict[WebSocket, dict] = {}        # socket -> {room, nickname, color}

COLORS = ["#00e5a0", "#7c5cfc", "#ff6b6b", "#ffd700", "#00b4d8", "#ff85a1", "#82e0aa", "#f0a500"]
_color_idx = 0


def _next_color():
    global _color_idx
    c = COLORS[_color_idx % len(COLORS)]
    _color_idx += 1
    return c


async def _broadcast(room_id: str, message: dict, exclude: WebSocket = None):
    """Send a message to all clients in a room."""
    if room_id not in chat_rooms:
        return
    dead = []
    for ws in chat_rooms[room_id]:
        if ws == exclude:
            continue
        try:
            await ws.send_text(json.dumps(message))
        except Exception:
            dead.append(ws)
    for ws in dead:
        chat_rooms[room_id].discard(ws)
        chat_users.pop(ws, None)


@app.websocket("/ws/chat/{room_id}")
async def ws_chat(websocket: WebSocket, room_id: str):
    """
    Multi-user live chat room with real-time money detection.

    Client sends:
      { "type": "join", "nickname": "Akash" }
      { "type": "message", "text": "you owe me $25" }

    Server broadcasts to all clients in the room:
      { "type": "join", "nickname": "Akash", "color": "#00e5a0", "members": [...] }
      { "type": "message", "nickname": "Akash", "color": "#00e5a0", "text": "...",
        "venmo_detection": { is_money, confidence, trigger_type, detected_amount, latency_ms },
        "timestamp": "3:28 AM" }
    """
    await websocket.accept()

    # Initialize room
    if room_id not in chat_rooms:
        chat_rooms[room_id] = set()
    chat_rooms[room_id].add(websocket)
    color = _next_color()
    chat_users[websocket] = {"room": room_id, "nickname": "Anonymous", "color": color}

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "invalid JSON"}))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "join":
                nickname = msg.get("nickname", "Anonymous").strip()[:20] or "Anonymous"
                chat_users[websocket]["nickname"] = nickname
                members = [chat_users[ws]["nickname"] for ws in chat_rooms[room_id] if ws in chat_users]
                # Broadcast join to everyone
                await _broadcast(room_id, {
                    "type": "join",
                    "nickname": nickname,
                    "color": color,
                    "members": members,
                })
                # Send color assignment back to joining user
                await websocket.send_text(json.dumps({
                    "type": "self",
                    "nickname": nickname,
                    "color": color,
                    "members": members,
                }))

            elif msg_type == "message":
                text = msg.get("text", "").strip()
                if not text:
                    continue

                nickname = chat_users[websocket]["nickname"]
                ts = datetime.utcnow().strftime("%-I:%M %p") if os.name != 'nt' else datetime.utcnow().strftime("%#I:%M %p")
                msg_id = f"{nickname}_{int(time.time()*1000)}"

                # Broadcast message INSTANTLY (no waiting for model)
                await _broadcast(room_id, {
                    "type": "message",
                    "msg_id": msg_id,
                    "nickname": nickname,
                    "color": color,
                    "text": text,
                    "timestamp": ts,
                })

                # Run detection in background and broadcast result separately
                if model_state["model"] is not None:
                    async def _detect_and_broadcast(txt, rm, nk, cl, mid):
                        result = await asyncio.get_event_loop().run_in_executor(None, run_inference, txt)
                        if result["is_money"]:
                            await _broadcast(rm, {
                                "type": "detection",
                                "msg_id": mid,
                                "nickname": nk,
                                "color": cl,
                                "text": txt,
                                "venmo_detection": {
                                    "is_money":        result["is_money"],
                                    "confidence":      result["confidence"],
                                    "trigger_type":    result["trigger_type"],
                                    "direction":       result["direction"],
                                    "detected_amount": result["detected_amount"],
                                    "latency_ms":      result["latency_ms"],
                                },
                            })
                    asyncio.create_task(_detect_and_broadcast(text, room_id, nickname, color, msg_id))

            elif msg_type == "typing":
                nickname = chat_users[websocket]["nickname"]
                await _broadcast(room_id, {
                    "type": "typing",
                    "nickname": nickname,
                }, exclude=websocket)

    except WebSocketDisconnect:
        nickname = chat_users.get(websocket, {}).get("nickname", "Unknown")
        chat_rooms.get(room_id, set()).discard(websocket)
        chat_users.pop(websocket, None)
        members = [chat_users[ws]["nickname"] for ws in chat_rooms.get(room_id, set()) if ws in chat_users]
        await _broadcast(room_id, {
            "type": "leave",
            "nickname": nickname,
            "members": members,
        })
        logger.info(f"Chat user '{nickname}' left room '{room_id}'")


@app.get("/health")
async def health():
    """Health check — returns model version and current accuracy."""
    return {
        "status":      "ok",
        "device":      str(DEVICE),
        "model_dir":   str(MODEL_DIR),
        "loaded_at":   model_state["loaded_at"],
        "version":     model_state["version"],
        "threshold":   CONFIDENCE_THRESHOLD,
        "uptime_reqs": stats["requests"],
    }


@app.get("/metrics")
async def metrics():
    """Live inference metrics."""
    return {
        "requests":        stats["requests"],
        "money_detected":  stats["money_detected"],
        "detection_rate":  round(stats["money_detected"] / max(stats["requests"], 1), 4),
        "avg_latency_ms":  round(stats["avg_latency_ms"], 2),
        "started_at":      stats["started_at"],
    }


@app.post("/reload")
async def reload_model():
    """Hot-reload the model from disk (used by continuous learning scheduler)."""
    try:
        load_model()
        return {"status": "ok", "loaded_at": model_state["loaded_at"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Demo UI ──
DEMO_DIR = Path(os.getenv("DEMO_DIR", str(Path(__file__).resolve().parent.parent / "demo")))
DEMO_HTML = DEMO_DIR / "paychat_demo.html"


@app.get("/")
async def serve_demo():
    """Serve the PayChat demo UI."""
    if DEMO_HTML.exists():
        return FileResponse(DEMO_HTML, media_type="text/html")
    return {"message": "PayChat Money Detection API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
