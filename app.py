import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from google import genai 
from google.genai import types
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor
from gtts import gTTS
from contextlib import closing
import asyncio
import edge_tts
import csv
import json

# Muat variabel dari file .env
load_dotenv()

app = Flask(__name__)

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# --- INISIALISASI GEMINI ---
client = None
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    print("✅ Koneksi Gemini API berhasil.")
except Exception as e:
    print(f"❌ Error Gemini API: {e}")


def call_gemini(contents, thinking_level="MINIMAL"):
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
    )
    return client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=config,
    )

KNOWLEDGE_BASE = {}
try:
    # Membaca file CSV saat server pertama kali nyala (biar enteng)
    with open('Dataset_RAG_Englishv2.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Ambil nama bahasa inggrisnya dan jadikan huruf kecil semua
            nama_inggris = row['English Name'].strip().lower()
            KNOWLEDGE_BASE[nama_inggris] = {
                'deskripsi': row['Simple Description (Context for AI)'],
                'kalimat_lks': row['Example Sentence (from LKS)']
            }
    print(f"✅ RAG Berhasil dimuat: {len(KNOWLEDGE_BASE)} materi LKS siap digunakan.")
except Exception as e:
    print(f"⚠️ File materi_lks.csv tidak ditemukan atau error: {e}")

# --- FUNGSI HELPER DATABASE ---
# --- FUNGSI HELPER DATABASE (UPDATE BUAT NEON) ---
def get_db_connection():
    # Langsung tembak pakai DATABASE_URL dari .env
    return psycopg2.connect(os.getenv("DATABASE_URL"))

# --- FUNGSI HELPER TTS KE BASE64 (BARU!) ---
# --- FUNGSI HELPER TTS NEURAL (EDGE-TTS) ---
def generate_audio_base64(text):
    try:
        # Pilihan Suara Guru: 
        # "en-US-AriaNeural" (Cewek dewasa, ramah)
        # "en-US-AnaNeural" (Cewek ceria, cocok buat anak kecil)
        # "en-US-GuyNeural" (Cowok)
        voice = "en-CA-ClaraNeural" 
        
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
        Contoh: 'book', 'lamp', 'eraser'. jika ada mascot guru dan papan tulis di gambar abaikan saja itu hanya 3d model virtual fokus identifikasi objek yang ada di atas marker aja.
        dan jawab kata bendanya secara umum saja misalnya phone charger menjadi charger, dll"
        """
        
        response = call_gemini(contents=[image, prompt], thinking_level="MINIMAL")
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

    object_name = str(data['object_name']).strip().lower()
    question_key = data['question_key']
    custom_question = data.get('custom_question', '')

    # --- CEK CACHE DATABASE (HEMAT API GEMINI) ---
    if question_key != "custom":
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM objects WHERE object_name = %s", (object_name,))
                    db_result = cur.fetchone()
                    
                    # Kalau baris bendanya ada, DAN jawaban untuk tombol ini udah pernah diisi...
                    if db_result and question_key in db_result and db_result[question_key]:
                        print(f"✅ BINGO! Jawaban {question_key} untuk {object_name} diambil dari DATABASE NEON!")
                        # Return dari DB + Suara langsung, STOP di sini, gak usah panggil Gemini
                        audio_b64 = generate_audio_base64(db_result[question_key])
                        return jsonify({"status": "sukses", "jawaban": db_result[question_key], "audio_base64": audio_b64})
        except Exception as db_error:
            print(f"⚠️ Gagal cek cache database: {db_error}")

    # --- 2. SIAPKAN PROMPT GEMINI ---
    print(f"🤖 Memanggil AI Gemini untuk menjawab {question_key} dari {object_name}...")

    # --- 2. SIAPKAN PROMPT / JAWABAN ---
    prompt = ""
    jawaban_ai = ""

# --- CEK APAKAH BENDA ADA DI BUKU LKS (RAG SYSTEM) ---
    data_lks = None
    if object_name in KNOWLEDGE_BASE:
        data_lks = KNOWLEDGE_BASE[object_name]

    # --- PROMPT "STRICT TEACHER" MODE (SUPER NATURAL) ---
    base_instruction = (
        f"You are a friendly but strict 4th-grade English teacher.\n"
        f"1. ANSWER ONLY IN 1 SHORT SENTENCE (maximum 10 words). No greeting, no intro.\n"
        f"2. ALWAYS USE ENGLISH. Never reply in Indonesian.\n"
        f"3. NEVER mention 'According to the data', 'Database', 'The text says', or 'I don't have information'. Just answer directly like a real human teacher.\n"
    )

    if question_key == "custom":
        if not custom_question:
            return jsonify({"status": "gagal", "pesan": "Pertanyaan manual kosong"}), 400
        
        context_str = f"Fact about {object_name}: {data_lks['deskripsi']}\n" if data_lks else ""
        
        prompt = (f"{base_instruction}"
                  f"4. The student must ask only about '{object_name}'.\n"
                  f"5. If the question is unrelated to '{object_name}' (for example politics, celebrities, math, or random world facts), reply EXACTLY: 'Sorry, I can only answer questions about {object_name}.'\n"
                  f"6. If the question is related to '{object_name}', answer briefly. If the Fact helps, use it. If the Fact is not enough, you may use your own general knowledge about '{object_name}'.\n"
                  f"{context_str}"
                  f"Student Question: {custom_question}\n"
                  f"Teacher's Answer (1 short sentence):")

    elif question_key == "definisi":
        # Jika objek ada di RAG, jawaban wajib mengacu pada fact RAG.
        if data_lks:
            fact = data_lks['deskripsi']
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is this?'.\n"
                      f"5. You MUST answer only from the Fact below. Do not add outside information.\n"
                      f"6. If the Fact is insufficient, reply exactly: 'I only know basic info about this object.'\n"
                      f"Fact: {fact}\n"
                      f"Teacher's Answer:")
        else:
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is this?'.\n"
                      f"5. This object is not in the RAG dataset, so use your general knowledge.\n"
                      f"Teacher's Answer:")

    elif question_key == "fungsi":
        if data_lks:
            fact = data_lks['deskripsi']
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is it for?'.\n"
                      f"5. You MUST answer only from the Fact below. Do not add outside information.\n"
                      f"6. If the Fact is insufficient, reply exactly: 'I only know basic info about this object.'\n"
                      f"Fact: {fact}\n"
                      f"Teacher's Answer:")
        else:
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is it for?'.\n"
                      f"5. This object is not in the RAG dataset, so use your general knowledge.\n"
                      f"Teacher's Answer:")

    elif question_key == "kalimat":
        # Untuk kalimat, jika ada di RAG pakai data LKS langsung biar mutlak.
        if data_lks and data_lks.get('kalimat_lks'):
            jawaban_ai = data_lks['kalimat_lks'].strip()
        else:
            prompt = (f"{base_instruction}"
                      f"4. Make a simple 4th-grade English sentence using the word '{object_name}'.\n"
                      f"Teacher's Answer:")
                  
    elif question_key == "ejaan":
        letters = [ch.upper() for ch in object_name if ch.isalpha()]
        if letters:
            jawaban_ai = " ".join([f"{letter}." for letter in letters])
        else:
            prompt = f"Spell the word '{object_name}' letter by letter. Separate each letter with a period. Example for 'BOOK': B. O. O. K."
    else:
        return jsonify({"status": "gagal", "pesan": "Kunci pertanyaan salah"}), 400

    try:
        if not jawaban_ai:
            response = call_gemini(contents=prompt, thinking_level="MINIMAL")
            jawaban_ai = (response.text or "").strip()

        if not jawaban_ai:
            jawaban_ai = "I only know basic info about this object."

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
        response = call_gemini(contents=[image, prompt], thinking_level="MINIMAL")
        jawaban_ai_text = (response.text or "").strip()

        if not jawaban_ai_text:
            jawaban_ai_text = "Sorry, I don't know how to answer that."

        # --- TAMBAHAN SUARA BUAT GAMBAR MANUAL ---
        audio_b64 = generate_audio_base64(jawaban_ai_text)

        # Balikannya sekarang ada audio_base64
        return jsonify({"status": "sukses", "jawaban": jawaban_ai_text, "audio_base64": audio_b64})

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 1. API UNTUK AMBIL DAFTAR BENDA (BUAT MENU QUIZ) ---
@app.route('/list-objects', methods=['GET'])
def list_objects():
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor() as cur:
                # Ambil semua nama benda yang pernah di-scan
                cur.execute("SELECT DISTINCT object_name FROM objects")
                rows = cur.fetchall()
                # Ubah jadi list simple
                object_list = [row[0] for row in rows] 
                return jsonify({"status": "sukses", "objects": object_list})
    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 2. API UNTUK GENERATE / AMBIL SOAL QUIZ ---
# --- 2. API UNTUK GENERATE / AMBIL SOAL QUIZ ---
@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.get_json()
    if not data or 'object_name' not in data:
        return jsonify({"status": "gagal", "pesan": "Data tidak lengkap"}), 400

    object_name = data['object_name'].lower()

    # A. CEK DATABASE DULU (SIAPA TAU UDAH PERNAH DIBIKIN)
    try:
        with closing(get_db_connection()) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT questions_json FROM quizzes WHERE object_name = %s", (object_name,))
                result = cur.fetchone()
                
                if result:
                    print(f"✅ Quiz untuk {object_name} diambil dari DATABASE NEON!")
                    return jsonify({"status": "sukses", "data": json.loads(result[0])})
    except Exception as e:
        print(f"⚠️ Gagal cek cache database quiz: {e}")

    # B. KALAU BELUM ADA, MINTA GEMINI BUATKAN
    print(f"🤖 Meminta Gemini membuat 10 Soal Quiz untuk: {object_name}...")
    
    # --- PROMPT BARU YANG LEBIH KETAT DAN SUPER GAMPANG ---
    prompt = (
        f"Create a text-only multiple-choice quiz about '{object_name}' for 4th-grade elementary students in Indonesia who are beginners in English.\n"
        f"Generate exactly 10 questions.\n"
        f"STRICT OUTPUT FORMAT: Return ONLY a raw JSON array. Do not use Markdown blocks (```json).\n"
        f"Format Structure:\n"
        f"[\n"
        f"  {{ \"question\": \"Where do you usually find a {object_name}?\", \"options\": [\"A) Option1\", \"B) Option2\", \"C) Option3\", \"D) Option4\"], \"correct_index\": 0 }}\n"
        f"]\n"
        f"Rules you MUST follow:\n"
        f"1. NO IMAGE REFERENCES (CRITICAL): The quiz is TEXT-ONLY. NEVER use phrases like 'in the picture', 'look at this image', 'what color is this', or 'in this photo'.\n"
        f"2. QUESTION TYPES: Make logical questions based on function or location. Examples: 'Where do you put a {object_name}?', 'We use a {object_name} to...', or 'What is inside a {object_name}?'.\n"
        f"3. EXTREMELY SIMPLE ENGLISH: Use basic vocabulary. Max 10 words per question.\n"
        f"4. SHORT OPTIONS: Options must be very short (1 to 4 words max).\n"
        f"5. MANDATORY PREFIX: Every single option string MUST start with exactly 'A) ', 'B) ', 'C) ', and 'D) '.\n"
        f"6. 'correct_index' is an integer: 0 for A, 1 for B, 2 for C, 3 for D."
    )

    try:
        response = call_gemini(contents=prompt, thinking_level="HIGH")
        raw_text = (response.text or "").strip()
        
        # Bersihkan "sampah" format dari Gemini (kalau dia bandel ngasih ```json)
        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        quiz_data = json.loads(raw_text) # Ubah teks jadi JSON

        # C. SIMPAN KE DATABASE (Biar besok gak mikir lagi)
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor() as cur:
                    json_str = json.dumps(quiz_data)
                    cur.execute(
                        "INSERT INTO quizzes (object_name, questions_json) VALUES (%s, %s) ON CONFLICT (object_name) DO NOTHING",
                        (object_name, json_str)
                    )
                    conn.commit()
                    print(f"💾 Quiz {object_name} berhasil disimpan ke Database!")
        except Exception as db_e:
            print(f"⚠️ Gagal simpan quiz ke DB: {db_e}")

        return jsonify({"status": "sukses", "data": quiz_data})

    except Exception as e:
        print(f"❌ Error API Quiz: {e}")
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)