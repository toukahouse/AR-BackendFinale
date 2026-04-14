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
import re

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


def is_related_custom_question(object_name, question_text):
    obj = str(object_name or "").strip().lower()
    q = " ".join(str(question_text or "").strip().lower().split())
    if not obj or not q:
        return False

    # Strong allow: explicit object mention.
    if obj in q:
        return True

    # Allow partial mention for multi-word objects, e.g. "remote" for "remote control".
    obj_parts = [p for p in obj.replace("-", " ").split() if len(p) >= 4]
    if any(part in q for part in obj_parts):
        return True

    intent_keywords = [
        "shape", "color", "size", "material", "function", "use", "used", "price", "buy", "store", "where",
        "bentuk", "warna", "ukuran", "bahan", "fungsi", "kegunaan", "harga", "beli", "toko", "dimana", "di mana"
    ]
    contextual_refs = ["it", "this", "that", "this object", "benda ini", "itu", "ini"]
    if any(k in q for k in intent_keywords) and any(r in q for r in contextual_refs):
        return True

    # Strong block for obvious out-of-topic world questions when object not mentioned.
    off_topic = [
        "president", "prime minister", "chancellor", "germany", "jerman", "celebrity", "actor", "football", "math",
        "politik", "pemerintah", "negara", "ibukota", "capital city", "planet", "history", "sejarah"
    ]
    if any(k in q for k in off_topic):
        return False

    # Fallback semantic check with Gemini in bilingual mode.
    classify_prompt = (
        "You are a classifier.\n"
        "Task: Decide if the user question is related to the scanned object.\n"
        "Object: " + obj + "\n"
        "Question: " + q + "\n"
        "Rules:\n"
        "- Related if asking attributes, function, usage, place to buy, care, parts, examples, spelling, or sentence about that object.\n"
        "- Question can be in English or Indonesian.\n"
        "- Unrelated if about politics, celebrities, random world facts, or another object.\n"
        "Output exactly one word: RELATED or UNRELATED."
    )

    try:
        resp = call_gemini(contents=classify_prompt, thinking_level="MINIMAL")
        label = (resp.text or "").strip().upper()
        return label.startswith("RELATED")
    except Exception:
        return False

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
                'kalimat_lks': row['Example Sentence (from LKS)'],
                'qna_lks': row.get('Asking and Giving Information', '')
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
                    if db_result and question_key in db_result and db_result[question_key]:
                        print(f"✅ BINGO! Jawaban {question_key} untuk {object_name} diambil dari DATABASE NEON!")
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

        if not is_related_custom_question(object_name, custom_question):
            blocked_answer = f"Sorry, I can only answer questions about {object_name}."
            audio_b64 = generate_audio_base64(blocked_answer)
            return jsonify({"status": "sukses", "jawaban": blocked_answer, "audio_base64": audio_b64})
        
        context_str = f"Fact about {object_name}: {data_lks['deskripsi']}\n" if data_lks else ""
        
        prompt = (f"{base_instruction}"
                  f"4. The student question is already confirmed related to '{object_name}'.\n"
                  f"5. Answer briefly. If the Fact helps, use it. If the Fact is not enough, you may use your own general knowledge about '{object_name}'.\n"
                  f"{context_str}"
                  f"Student Question: {custom_question}\n"
                  f"Teacher's Answer (1 short sentence):")

    elif question_key == "definisi":
        if data_lks:
            fact = data_lks['deskripsi']
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is this?'.\n"
                      f"5. You MUST answer using the Fact below as your main source. Do not contradict it.\n"
                      f"6. Mention the object name and its shape or physical characteristics in one short sentence.\n"
                      f"7. If the Fact is insufficient, reply exactly: 'I only know basic info about this object.'\n"
                      f"Fact: {fact}\n"
                      f"Teacher's Answer:")
        else:
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is this?' about '{object_name}'.\n"
                      f"5. This object is not in the RAG dataset, so use your general knowledge about '{object_name}'.\n"
                      f"6. Mention the object name and one simple physical characteristic.\n"
                      f"Teacher's Answer:")

    elif question_key == "fungsi":
        if data_lks:
            fact = data_lks['deskripsi']
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is it for?'.\n"
                      f"5. You MUST answer using the Fact below as your main source. Do not contradict it.\n"
                      f"6. Explain the main use of the object in one short sentence.\n"
                      f"7. If the Fact is insufficient, reply exactly: 'I only know basic info about this object.'\n"
                      f"Fact: {fact}\n"
                      f"Teacher's Answer:")
        else:
            prompt = (f"{base_instruction}"
                      f"4. The student asks 'What is it for?' about '{object_name}'.\n"
                      f"5. This object is not in the RAG dataset, so use your general knowledge about '{object_name}'.\n"
                      f"6. Answer the object's main use in simple English.\n"
                      f"Teacher's Answer:")

    elif question_key == "kalimat":
        if data_lks and data_lks.get('kalimat_lks'):
            prompt = (f"{base_instruction}"
                      f"4. Make one simple 4th-grade English sentence using the word '{object_name}'.\n"
                      f"5. Use the reference below as your main source, but rewrite it naturally.\n"
                      f"Reference: {data_lks['kalimat_lks']}\n"
                      f"Teacher's Answer:")
        else:
            prompt = (f"{base_instruction}"
                      f"4. Make a simple 4th-grade English sentence using the word '{object_name}'.\n"
                      f"Teacher's Answer:")
                  
    elif question_key == "ejaan":
        prompt = (
            f"Spell the word '{object_name}' letter by letter. "
            f"Separate each letter with a period. Example for 'BOOK': B. O. O. K."
        )
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

# --- 5. ENDPOINT KHUSUS BUAT BACAIN SOAL KUIS ---
@app.route('/tts-soal', methods=['POST'])
def tts_soal():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"status": "gagal", "pesan": "Butuh parameter 'text'"}), 400

    text = data['text']
    print(f"🔊 Generate Voice Soal: {text}")
    
    # Langsung pakai fungsi helper Edge-TTS yang udah ada di kodemu!
    audio_b64 = generate_audio_base64(text)

    if audio_b64:
        return jsonify({"status": "sukses", "audio_base64": audio_b64})
    else:
        return jsonify({"status": "gagal", "pesan": "Gagal generate audio"}), 500

# --- 2. API UNTUK GENERATE / AMBIL SOAL QUIZ ---
@app.route('/generate-quiz', methods=['POST'])
def generate_quiz():
    data = request.get_json()
    if not data or 'object_name' not in data:
        return jsonify({"status": "gagal", "pesan": "Data tidak lengkap"}), 400

    object_name = str(data['object_name']).strip().lower()
    force_regenerate = bool(data.get('force_regenerate', False))
    cached_quiz = None

    # A. CEK DATABASE DULU (SIAPA TAU UDAH PERNAH DIBIKIN)
    if not force_regenerate:
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT questions_json FROM quizzes WHERE object_name = %s", (object_name,))
                    result = cur.fetchone()

                    if result:
                        try:
                            cached_quiz = json.loads(result[0])
                        except Exception:
                            cached_quiz = None
                            print(f"⚠️ Cache quiz untuk {object_name} rusak, akan regenerate.")
        except Exception as e:
            print(f"⚠️ Gagal cek cache database quiz: {e}")

    rag_data = KNOWLEDGE_BASE.get(object_name)
    rag_context = ""
    if rag_data:
        rag_context = (
            f"RAG Fact - Description: {rag_data.get('deskripsi', '')}\n"
            f"RAG Fact - Example Sentence: {rag_data.get('kalimat_lks', '')}\n"
            f"RAG Fact - QnA: {rag_data.get('qna_lks', '')}\n"
        )

    ambiguous_option_groups = [
        {"pen", "pencil", "marker", "crayon", "chalk"},
        {"book", "notebook"},
        {"sofa", "couch"},
        {"phone", "smartphone", "mobile phone", "cell phone"},
        {"cup", "mug", "glass"},
    ]

    off_topic_keywords = [
        "president", "prime minister", "germany", "jerman", "planet", "history", "sejarah",
        "celebrity", "football", "chancellor", "capital city", "politik", "pemerintah"
    ]

    def _build_quiz_prompt(excluded_questions=None):
        excluded_questions = excluded_questions or []
        excluded_block = ""
        if excluded_questions:
            excluded_block = (
                "Do not generate any of these question texts again:\n"
                + "\n".join([f"- {q}" for q in excluded_questions])
                + "\n"
            )

        rag_rules = ""
        if rag_data:
            rag_rules = (
                "RAG POLICY:\n"
                "- Use the RAG facts below as primary source.\n"
                "- At least 6 out of 10 questions must be directly answerable from these RAG facts.\n"
                "- If RAG Fact - QnA exists, create at least 2 questions inspired by that QnA pattern.\n"
                "- If using location information from RAG, ask concrete place-choice questions like 'Where is the charger?' instead of yes/no questions.\n"
                "- The remaining questions may use simple common knowledge, but still must stay about the same object.\n"
                "- Never contradict RAG facts.\n"
                f"{rag_context}\n"
            )
        else:
            rag_rules = (
                "RAG POLICY:\n"
                "- This object is not found in RAG dataset.\n"
                "- Use simple, safe general knowledge about the object.\n"
                "- Keep quality and difficulty at 4th-grade beginner level.\n"
            )

        return (
            f"Create a text-only multiple-choice quiz about '{object_name}' for 4th-grade elementary students in Indonesia who are beginners in English.\n"
            f"Generate exactly 10 questions.\n"
            f"STRICT OUTPUT FORMAT: Return ONLY a raw JSON array. Do not use Markdown blocks (```json).\n"
            f"Format Structure:\n"
            f"[\n"
            f"  {{ \"question\": \"Where do you usually find a {object_name}?\", \"options\": [\"A) Option1\", \"B) Option2\", \"C) Option3\", \"D) Option4\"], \"correct_index\": 0 }}\n"
            f"]\n"
            f"Rules you MUST follow:\n"
            f"1. NO IMAGE REFERENCES: The quiz is TEXT-ONLY.\n"
            f"2. UNIQUE QUESTIONS: All 10 question texts must be different in meaning and wording.\n"
            f"3. EXTREMELY SIMPLE ENGLISH: Max 8 words per question.\n"
            f"4. SHORT OPTIONS: Each option is 1 to 3 words only.\n"
            f"5. MANDATORY PREFIX: options must start with exactly 'A) ', 'B) ', 'C) ', 'D) '.\n"
            f"6. correct_index is integer 0..3 matching correct option.\n"
            f"7. Keep questions answerable by kids (no tricky/ambiguous wording).\n"
            f"8. Every question must have exactly one clearly correct answer. Avoid questions that can have multiple logical answers in real life.\n"
            f"9. DO NOT make yes/no questions like 'Is this in the living room?' or 'Can it be on a table?'.\n"
            f"10. Include at least 2 sentence-completion questions using exactly one blank: '....'. Example: 'I use a .... to charge my phone.'\n"
            f"11. Include at least 1 translation question from Indonesian to English. Example pattern: What is \"meja\" in English?\n"
            f"12. For that translation question, the correct option must be '{object_name}' (or same word with article/plural form).\n"
            f"13. Prefer these question types: function, part, material, place, sentence completion, translation (Indonesian-to-English), and simple vocabulary.\n"
            f"14. Every non-blank and non-translation question must mention '{object_name}' (or its clear short form).\n"
            f"15. Never use generic templates like 'What do you use to write?'. Use object-focused wording such as 'What is a pen for?'.\n"
            f"16. Wrong options must be clearly wrong for kids. Never put two options that can both be true in daily life.\n"
            f"17. Keep all questions on-topic about '{object_name}', never random world facts.\n"
            f"{rag_rules}"
            f"{excluded_block}"
        )

    def _normalize_question_text(text):
        return " ".join(str(text).strip().lower().split())

    def _normalize_option_text(text):
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(text).strip().lower())
        return " ".join(cleaned.split())

    def _option_matches_object(option_text):
        obj = _normalize_option_text(object_name)
        opt = _normalize_option_text(option_text)
        if not obj or not opt:
            return False
        if opt == obj:
            return True
        if opt.endswith("s") and opt[:-1] == obj:
            return True
        return obj in opt

    def _question_mentions_object(question_text):
        obj = _normalize_question_text(object_name)
        q = _normalize_question_text(question_text)
        if not obj or not q:
            return False
        if obj in q:
            return True
        obj_parts = [part for part in obj.replace("-", " ").split() if len(part) >= 4]
        return any(part in q for part in obj_parts)

    def _is_ambiguous_yes_no_question(text):
        normalized = _normalize_question_text(text)
        blocked_starts = (
            "is ", "are ", "can ", "do ", "does ", "did ", "was ", "were ", "has ", "have "
        )
        return normalized.startswith(blocked_starts)

    def _is_generic_function_question(text):
        normalized = _normalize_question_text(text)
        generic_patterns = (
            "what do you use to",
            "what can you use to",
            "what is used to",
            "which tool do you use to",
            "what do we use to",
        )
        return any(pattern in normalized for pattern in generic_patterns)

    def _is_translation_question(text):
        normalized = _normalize_question_text(text)
        if not normalized.startswith("what is "):
            return False
        return " in english" in normalized

    def _is_off_topic_question(text):
        normalized = _normalize_question_text(text)
        return any(keyword in normalized for keyword in off_topic_keywords)

    def _validate_quiz_payload(items):
        if not isinstance(items, list):
            return False
        if len(items) != 10:
            return False

        seen_questions = set()
        sentence_completion_count = 0
        translation_question_count = 0
        for item in items:
            if not isinstance(item, dict):
                return False
            if "question" not in item or "options" not in item or "correct_index" not in item:
                return False

            q_text = _normalize_question_text(item["question"])
            if not q_text or q_text in seen_questions:
                return False
            if _is_ambiguous_yes_no_question(q_text):
                return False
            if _is_off_topic_question(q_text):
                return False
            if len(q_text.replace("....", " ").split()) > 8:
                return False
            seen_questions.add(q_text)

            has_blank = "...." in str(item["question"])
            is_translation = _is_translation_question(item["question"])
            if has_blank:
                sentence_completion_count += 1
            elif is_translation:
                translation_question_count += 1
            elif not _question_mentions_object(q_text):
                return False

            options = item["options"]
            if not isinstance(options, list) or len(options) != 4:
                return False
            expected_prefix = ["A) ", "B) ", "C) ", "D) "]
            normalized_options = []
            for idx, opt in enumerate(options):
                if not isinstance(opt, str) or not opt.startswith(expected_prefix[idx]):
                    return False
                option_text = _normalize_option_text(opt[3:])
                if not option_text:
                    return False
                if len(option_text.split()) > 3:
                    return False
                normalized_options.append(option_text)

            if len(set(normalized_options)) != 4:
                return False

            if _is_generic_function_question(q_text):
                if not _question_mentions_object(q_text):
                    return False
                for group in ambiguous_option_groups:
                    group_hits = sum(1 for opt in normalized_options if opt in group)
                    if group_hits >= 2:
                        return False

            correct_index = item["correct_index"]
            if not isinstance(correct_index, int) or correct_index < 0 or correct_index > 3:
                return False

            if is_translation:
                correct_option = normalized_options[correct_index]
                if not _option_matches_object(correct_option):
                    return False

        if sentence_completion_count < 2:
            return False

        if translation_question_count < 1:
            return False

        return True

    if not force_regenerate and cached_quiz:
        if _validate_quiz_payload(cached_quiz):
            print(f"✅ Quiz untuk {object_name} diambil dari DATABASE NEON!")
            return jsonify({"status": "sukses", "data": cached_quiz})
        print(f"⚠️ Cache quiz lama untuk {object_name} tidak lolos validasi terbaru, regenerate.")

    # B. KALAU BELUM ADA, MINTA GEMINI BUATKAN
    print(f"🤖 Meminta Gemini membuat 10 Soal Quiz untuk: {object_name}...")

    try:
        quiz_data = None
        excluded_questions = []

        for _ in range(5):
            prompt = _build_quiz_prompt(excluded_questions)
            response = call_gemini(contents=prompt, thinking_level="MINIMAL")
            raw_text = (response.text or "").strip()

            # Bersihkan "sampah" format dari Gemini (kalau dia bandel ngasih ```json)
            if raw_text.startswith("```"):
                raw_text = raw_text.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(raw_text)
            except Exception:
                excluded_questions = []
                continue

            if _validate_quiz_payload(parsed):
                quiz_data = parsed
                break

            if isinstance(parsed, list):
                excluded_questions = [str(item.get("question", "")).strip() for item in parsed if isinstance(item, dict)]

        if not quiz_data:
            return jsonify({
                "status": "gagal",
                "pesan": "AI gagal membuat quiz valid dan unik. Coba lagi."
            }), 500

        # C. SIMPAN KE DATABASE (Biar besok gak mikir lagi)
        try:
            with closing(get_db_connection()) as conn:
                with conn.cursor() as cur:
                    json_str = json.dumps(quiz_data)
                    cur.execute(
                        "INSERT INTO quizzes (object_name, questions_json) VALUES (%s, %s) ON CONFLICT (object_name) DO UPDATE SET questions_json = EXCLUDED.questions_json",
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
    app.run(host='0.0.0.0', port=5001, debug=True)