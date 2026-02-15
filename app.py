import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from google import genai 
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor
from gtts import gTTS
from contextlib import closing
import asyncio
import edge_tts

# Muat variabel dari file .env
load_dotenv()

app = Flask(__name__)

# --- INISIALISASI GEMINI ---
client = None
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Koneksi Gemini API berhasil.")
except Exception as e:
    print(f"❌ Error Gemini API: {e}")

# --- FUNGSI HELPER DATABASE ---
def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS")
    )

# --- FUNGSI HELPER TTS KE BASE64 (BARU!) ---
# --- FUNGSI HELPER TTS NEURAL (EDGE-TTS) ---
def generate_audio_base64(text):
    try:
        # Pilihan Suara Guru: 
        # "en-US-AriaNeural" (Cewek dewasa, ramah)
        # "en-US-AnaNeural" (Cewek ceria, cocok buat anak kecil)
        # "en-US-GuyNeural" (Cowok)
        voice = "en-US-AriaNeural" 
        
        # Karena edge-tts itu asynchronous, kita bungkus pakai asyncio
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data

        # Jalankan dan tangkap hasil byte audio-nya
        audio_bytes = asyncio.run(_generate())
        
        # Ubah ke Base64 buat dikirim ke Unity
        return base64.b64encode(audio_bytes).decode('utf-8')
    except Exception as e:
        print(f"⚠️ Error generate Neural TTS: {e}")
        return ""

@app.route('/')
def index():
    return "🚀 Backend AR Skripsi Nova Ready!"

# --- 1. ENDPOINT TEXT-TO-SPEECH (Tetap dipertahankan) ---
@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"status": "gagal", "pesan": "Butuh parameter 'text'"}), 400

    try:
        tts = gTTS(text=data['text'], lang='en', slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return send_file(mp3_fp, mimetype="audio/mpeg", as_attachment=False, download_name="speech.mp3")
    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 2. ENDPOINT IDENTIFIKASI OBJEK ---
@app.route('/identifikasi-objek', methods=['POST'])
def identifikasi_objek():
    image = None
    if 'file' in request.files:
        image = Image.open(request.files['file'].stream)
    elif request.is_json and 'image_base64' in request.get_json():
        image_data = base64.b64decode(request.get_json()['image_base64'])
        image = Image.open(io.BytesIO(image_data))
    else:
        return jsonify({"status": "gagal", "pesan": "Kirim file gambar atau JSON image_base64"}), 400

    try:
        prompt = """
        Kamu adalah API backend untuk sebuah aplikasi edukasi AR Bahasa Inggris.
        Tugasmu adalah mengidentifikasi benda di kamar tidur atau ruang tamu.
        Fokus HANYA pada objek yang diletakkan DI ATAS marker. 
        Jika ada tulisan "taruh benda di sini" terlihat sangat jelas tanpa tertutup benda, jawab "unknown".
        Abaikan background. Balas HANYA dengan nama objek dalam Bahasa Inggris (tunggal).
        Contoh: 'book', 'lamp', 'eraser'.
        dan Use simple words. Add commas (,) frequently to create natural reading pauses."
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[image, prompt],
        )
        object_name = (response.text or "").strip().lower()

        audio_b64 = "" # Variabel kosong buat suara

        if object_name and object_name != "unknown":
            # --- TAMBAHAN SUARA PAS SCAN ---
            # Si Guru bakal ngomong: "I see a book!"
            audio_b64 = generate_audio_base64(f"I see a {object_name}")

            try:
                with closing(get_db_connection()) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO objects (object_name) VALUES (%s) ON CONFLICT (object_name) DO NOTHING",
                            (object_name,)
                        )
                        conn.commit()
            except Exception as db_error:
                print(f"⚠️ DB Error: {db_error}")

        # Balikannya sekarang ada audio_base64
        return jsonify({
            "status": "sukses", 
            "object_name": object_name,
            "audio_base64": audio_b64
        })

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 3. ENDPOINT Q&A TEMPLATE ---
@app.route('/tanya-ai', methods=['POST'])
def tanya_ai():
    data = request.get_json()
    if not data or 'object_name' not in data or 'question_key' not in data:
        return jsonify({"status": "gagal", "pesan": "Data tidak lengkap"}), 400

    object_name = data['object_name']
    question_key = data['question_key']
    custom_question = data.get('custom_question', '')

    # --- CEK CACHE DATABASE ---
    if question_key != "custom":
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM objects WHERE object_name = %s", (object_name,))
                    db_result = cur.fetchone()
                    
                    if db_result and question_key in db_result and db_result[question_key]:
                        jawaban = db_result[question_key]
                        # --- TAMBAHAN SUARA DARI CACHE ---
                        audio_b64 = generate_audio_base64(jawaban)
                        return jsonify({"status": "sukses", "jawaban": jawaban, "audio_base64": audio_b64})
        except Exception as db_error:
            print(f"⚠️ Gagal cek cache: {db_error}")

    # --- 2. SIAPKAN PROMPT GEMINI ---
    prompt = ""
    question_map = {
        "definisi": f"What is a {object_name}?",
        "fungsi": f"What is a {object_name} for?",
        "ejaan": f"How do you spell '{object_name}'?",
        "kalimat": f"Make a simple sentence with '{object_name}'."
    }

    if question_key == "custom":
        if not custom_question:
            return jsonify({"status": "gagal", "pesan": "Pertanyaan manual kosong"}), 400
        prompt = (
            f"Context: You are a helpful teacher explaining a physical object to a 4th-grade student.\n"
            f"The student is holding an object named '{object_name}'.\n"
            f"STRICT RULE: Always interpret '{object_name}' as an INANIMATE OBJECT, TOOL, ELECTRONIC DEVICE, or SCHOOL SUPPLY.\n"
            f"NEVER interpret it as an animal, living creature, or person.\n"
            f"Examples:\n"
            f"- If object is 'Mouse', assume it is a Computer Mouse (NOT a rat).\n"
            f"- If object is 'Bat', assume it is a Baseball Bat (NOT a flying mammal).\n"
            f"- If object is 'Crane', assume it is a Machine (NOT a bird).\n"
            f"User's Question: '{custom_question}'\n"
            f"Instruction: Answer the question based on the definition above. Keep it short and simple (max 2 sentences)."
            f"Use simple words. Add commas (,) frequently to create natural reading pauses."
        )
    elif question_key in question_map:
# INI KODE BARU
        if question_key == "ejaan":
            prompt = f"Spell the word '{object_name}' letter by letter. Separate each letter with a period and a space. Example: B. O. O. K. Only reply with the spelling."
        else:
            prompt = f"Pertanyaan: '{question_map[question_key]}'. Jawab sangat singkat dalam Bahasa Inggris untuk anak 10 tahun (maks 2 kalimat). Jangan menyapa."
    else:
        return jsonify({"status": "gagal", "pesan": "Kunci pertanyaan salah"}), 400

    try:
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=prompt)
        jawaban_ai = (response.text or "").strip()

        # --- TAMBAHAN SUARA JAWABAN GEMINI ---
        audio_b64 = generate_audio_base64(jawaban_ai)

        if question_key != "custom":
            try:
                with closing(get_db_connection()) as conn:
                    with conn.cursor() as cur:
                        if question_key in ["definisi", "fungsi", "ejaan", "kalimat"]:
                            sql = f"UPDATE objects SET {question_key} = %s WHERE object_name = %s"
                            cur.execute(sql, (jawaban_ai, object_name))
                            conn.commit()
            except Exception as db_error:
                print(f"⚠️ Gagal simpan ke cache: {db_error}")

        # Balikannya sekarang ada audio_base64
        return jsonify({"status": "sukses", "jawaban": jawaban_ai, "audio_base64": audio_b64})

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 4. ENDPOINT TANYA MANUAL GAMBAR ---
@app.route('/tanya-gambar-manual', methods=['POST'])
def tanya_gambar_manual():
    if 'image_file' not in request.files or 'question_text' not in request.form:
        return jsonify({"status": "gagal", "pesan": "Kirim file gambar dan teks pertanyaan"}), 400

    try:
        image = Image.open(request.files['image_file'].stream)
        question_text = request.form['question_text']
        
        prompt = f"""
        Lihat gambar ini. Jawab pertanyaan siswa: "{question_text}"
        Jawab dengan Bahasa Inggris yang SANGAT SINGKAT (cocok untuk anak 10 tahun).
        Jangan menyapa, langsung jawabannya. Use simple words. Add commas (,) frequently to create natural reading pauses."
        """
        response = client.models.generate_content(model="gemini-2.5-flash-lite", contents=[image, prompt])
        jawaban_ai_text = (response.text or "").strip()

        if not jawaban_ai_text:
            jawaban_ai_text = "Sorry, I don't know how to answer that."

        # --- TAMBAHAN SUARA BUAT GAMBAR MANUAL ---
        audio_b64 = generate_audio_base64(jawaban_ai_text)

        # Balikannya sekarang ada audio_base64
        return jsonify({"status": "sukses", "jawaban": jawaban_ai_text, "audio_base64": audio_b64})

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)