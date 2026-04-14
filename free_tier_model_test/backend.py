import base64
import io
import os
import time

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("FREE_TIER_GEMINI_API_KEY") or "").strip()
MODEL_NAME = "gemma-4-31b-it"
DEFAULT_THINKING_LEVEL = os.getenv("FREE_TIER_THINKING_LEVEL", "HIGH").strip().upper()
MAX_IMAGE_SIZE = int(os.getenv("FREE_TIER_IMAGE_MAX_SIZE", "1280"))

app = Flask(__name__)

client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
        print(f"[free-tier-test] API siap. Model: {MODEL_NAME}")
    except Exception as exc:
        print(f"[free-tier-test] Gagal inisialisasi API client: {exc}")
else:
    print("[free-tier-test] GEMINI_API_KEY / FREE_TIER_GEMINI_API_KEY belum diisi.")


def _build_config(thinking_level=None):
    level = (thinking_level or DEFAULT_THINKING_LEVEL or "HIGH").strip().upper()
    return types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=level),
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",
            ),
        ],
    )


def _stream_generate(contents, thinking_level=None):
    if client is None:
        raise RuntimeError("GEMINI_API_KEY / FREE_TIER_GEMINI_API_KEY belum diisi atau client gagal dibuat.")

    config = _build_config(thinking_level=thinking_level)
    chunks = []

    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=config,
    ):
        text = getattr(chunk, "text", None)
        if text:
            chunks.append(text)

    return "".join(chunks).strip()


def _normalize_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    return image


def _read_image_from_request(req):
    if "image_file" in req.files:
        return _normalize_image(Image.open(req.files["image_file"].stream))

    if req.is_json:
        data = req.get_json(silent=True) or {}
        image_b64 = data.get("image_base64")
        if image_b64:
            raw_bytes = base64.b64decode(image_b64)
            return _normalize_image(Image.open(io.BytesIO(raw_bytes)))

    return None


@app.route("/")
def index():
    return jsonify(
        {
            "status": "ok",
            "service": "free-tier-model-test",
            "message": "Backend testing model aktif.",
            "model": MODEL_NAME,
        }
    )


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "service": "free-tier-model-test",
            "model": MODEL_NAME,
            "api_key_configured": bool(API_KEY),
        }
    )


@app.route("/chat", methods=["POST"])
def chat():
    if client is None:
        return jsonify(
            {
                "status": "gagal",
                "pesan": "GEMINI_API_KEY / FREE_TIER_GEMINI_API_KEY belum diisi atau client gagal dibuat.",
            }
        ), 500

    data = request.get_json(silent=True) or {}
    user_text = str(data.get("text", "")).strip()
    thinking_level = data.get("thinking_level")

    if not user_text:
        return jsonify({"status": "gagal", "pesan": "Field 'text' wajib diisi."}), 400

    start = time.time()
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            ),
        ]
        reply = _stream_generate(contents=contents, thinking_level=thinking_level)

        if not reply:
            reply = "(Model tidak mengembalikan teks balasan.)"

        return jsonify(
            {
                "status": "sukses",
                "reply": reply,
                "model": MODEL_NAME,
                "duration_seconds": round(time.time() - start, 2),
            }
        )
    except Exception as exc:
        return jsonify({"status": "gagal", "pesan": str(exc)}), 500


@app.route("/image-chat", methods=["POST"])
def image_chat():
    if client is None:
        return jsonify(
            {
                "status": "gagal",
                "pesan": "GEMINI_API_KEY / FREE_TIER_GEMINI_API_KEY belum diisi atau client gagal dibuat.",
            }
        ), 500

    image = _read_image_from_request(request)
    if image is None:
        return jsonify(
            {
                "status": "gagal",
                "pesan": "Kirim gambar lewat multipart key 'image_file' atau JSON 'image_base64'.",
            }
        ), 400

    if request.is_json:
        data = request.get_json(silent=True) or {}
        prompt = str(data.get("prompt", "")).strip()
        thinking_level = data.get("thinking_level")
    else:
        prompt = str(request.form.get("prompt", "")).strip()
        thinking_level = request.form.get("thinking_level")

    if not prompt:
        prompt = "Ini gambar apa? Jelaskan singkat dan jelas."

    start = time.time()
    try:
        reply = _stream_generate(contents=[image, prompt], thinking_level=thinking_level)

        if not reply:
            reply = "(Model tidak mengembalikan teks balasan.)"

        return jsonify(
            {
                "status": "sukses",
                "reply": reply,
                "model": MODEL_NAME,
                "duration_seconds": round(time.time() - start, 2),
            }
        )
    except Exception as exc:
        return jsonify({"status": "gagal", "pesan": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FREE_TIER_PORT", "5002"))
    app.run(host="127.0.0.1", port=port, debug=False)
