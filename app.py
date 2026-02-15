import os
import io
import base64
from gtts import gTTS
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from google import genai 
from PIL import Image
import psycopg2
from psycopg2.extras import RealDictCursor
from gtts import gTTS
from contextlib import closing # Biar database otomatis nutup

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

@app.route('/')
def index():
    return "🚀 Backend AR Skripsi Nova Ready!"

# --- 1. ENDPOINT TEXT-TO-SPEECH ---
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
        
        return send_file(
            mp3_fp,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="speech.mp3"
        )
    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 2. ENDPOINT IDENTIFIKASI OBJEK ---
@app.route('/identifikasi-objek', methods=['POST'])
def identifikasi_objek():
    image = None
    
    # Support upload via form (Postman) atau JSON Base64 (Unity)
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
        """
        
        # Peningkatan: Langsung kirim objek PIL Image tanpa upload ke file server!
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=[image, prompt],
        )
        
        object_name = (response.text or "").strip().lower()

        # Simpan ke Database dengan aman pakai contextlib (otomatis close)
        if object_name and object_name != "unknown":
            try:
                with closing(get_db_connection()) as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO objects (object_name) VALUES (%s) ON CONFLICT (object_name) DO NOTHING",
                            (object_name,)
                        )
                        conn.commit()
            except Exception as db_error:
                print(f"⚠️ DB Error (Abaikan jika tes lokal tanpa DB): {db_error}")

        return jsonify({"status": "sukses", "object_name": object_name})

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

# --- 3. ENDPOINT Q&A TEMPLATE ---
@app.route('/tanya-ai', methods=['POST'])
def tanya_ai():
    data = request.get_json()
    # Pastikan data dasar ada
    if not data or 'object_name' not in data or 'question_key' not in data:
        return jsonify({"status": "gagal", "pesan": "Data tidak lengkap"}), 400

    object_name = data['object_name']
    question_key = data['question_key']
    # Ambil pertanyaan manual (kalau ada), default string kosong
    custom_question = data.get('custom_question', '')

    # --- 1. CEK CACHE DATABASE (Hanya untuk Template, bukan Custom) ---
    if question_key != "custom":
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Pastikan kolom ada di tabel sebelum select (opsional, tapi aman)
                    cur.execute("SELECT * FROM objects WHERE object_name = %s", (object_name,))
                    db_result = cur.fetchone()
                    
                    # Cek apakah kolom key tersebut ada isinya
                    if db_result and question_key in db_result and db_result[question_key]:
                        return jsonify({"status": "sukses", "jawaban": db_result[question_key]})
        except Exception as db_error:
            print(f"⚠️ Gagal cek cache: {db_error}")

    # --- 2. SIAPKAN PROMPT GEMINI ---
    prompt = ""
    
    # Map untuk pertanyaan template
    question_map = {
        "definisi": f"What is a {object_name}?",
        "fungsi": f"What is a {object_name} for?",
        "ejaan": f"How do you spell '{object_name}'?",
        "kalimat": f"Make a simple sentence with '{object_name}'."
    }

    # Logika Pembuatan Prompt
    if question_key == "custom":
        # INI LOGIKA BARU BUAT TANYA MANUAL
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
        )
    elif question_key in question_map:
        # LOGIKA LAMA (TEMPLATE)
        if question_key == "ejaan":
            prompt = f"Eja kata '{object_name}' huruf per huruf, pisahkan dengan tanda hubung (-). Contoh: B-O-O-K. Balas ejaannya saja."
        else:
            prompt = f"Pertanyaan: '{question_map[question_key]}'. Jawab sangat singkat dalam Bahasa Inggris untuk anak 10 tahun (maks 2 kalimat). Jangan menyapa."
    else:
        # Kalau key ngaco
        return jsonify({"status": "gagal", "pesan": "Kunci pertanyaan salah"}), 400

    # --- 3. KIRIM KE GEMINI ---
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
        )
        jawaban_ai = (response.text or "").strip()

        # --- 4. SIMPAN KE DATABASE (Hanya untuk Template) ---
        # Kita ga simpan pertanyaan custom karena jawabannya variatif banget
        if question_key != "custom":
            try:
                with closing(get_db_connection()) as conn:
                    with conn.cursor() as cur:
                        # Pastikan nama kolom aman dari SQL Injection atau error
                        if question_key in ["definisi", "fungsi", "ejaan", "kalimat"]:
                            sql = f"UPDATE objects SET {question_key} = %s WHERE object_name = %s"
                            cur.execute(sql, (jawaban_ai, object_name))
                            conn.commit()
            except Exception as db_error:
                print(f"⚠️ Gagal simpan ke cache: {db_error}")

        return jsonify({"status": "sukses", "jawaban": jawaban_ai})

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
        Jangan menyapa, langsung jawabannya.
        """

        # Peningkatan: Langsung pakai objek PIL
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", # Samain ke 2.0 flash biar konsisten & cepet
            contents=[image, prompt],
        )
        jawaban_ai_text = (response.text or "").strip()

        if not jawaban_ai_text:
            jawaban_ai_text = "Sorry, I don't know how to answer that."

        return jsonify({"status": "sukses", "jawaban": jawaban_ai_text})

    except Exception as e:
        return jsonify({"status": "gagal", "pesan": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)