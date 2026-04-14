import requests
import concurrent.futures
import time

# Ganti dengan URL Cloudflare kamu
BASE_URL = "https://api.supernovaapp.dev"

def test_tanya_manual(id_siswa):
    url = f"{BASE_URL}/tanya-ai"
    
    # Kita bedain pertanyaan tiap siswa biar AI-nya beneran mikir 
    # (nggak bisa nyontek hasil sebelumnya)
    payload = {
        "object_name": "book",
        "question_key": "custom",
        "custom_question": f"Teacher, why is a book important for student number {id_siswa}?"
    }
    
    start_time = time.time()
    try:
        # Batas waktu nunggu AI ngetik dan ngomong (generate audio)
        response = requests.post(url, json=payload, timeout=40) 
        waktu = round(time.time() - start_time, 2)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'sukses':
                print(f"✅ Siswa {id_siswa} (Chat) Sukses! Waktu: {waktu} detik")
                # Kalau mau liat AI-nya jawab apa, buka comment di bawah ini:
                # print(f"   Jawaban AI: {data.get('jawaban')}")
                return True
            else:
                print(f"❌ Siswa {id_siswa} (Chat) Gagal AI! Pesan: {data.get('pesan')} | Waktu: {waktu} detik")
                return False
        else:
            print(f"❌ Siswa {id_siswa} (Chat) Gagal Server! HTTP Status: {response.status_code}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"⚠️ Siswa {id_siswa} (Chat) TIMEOUT! Server terlalu sibuk.")
        return False
    except Exception as e:
        print(f"⚠️ Siswa {id_siswa} (Chat) Error: {e}")
        return False

if __name__ == "__main__":
    # KITA SET 15 SISWA AJA BIAR GAK KENA LIMIT API GRATISAN GOOGLE
    JUMLAH_SISWA = 15

    print(f"🚀 MEMULAI STRESS TEST: {JUMLAH_SISWA} SISWA TANYA MANUAL BERSAMAAN...")
    start_total = time.time()

    # Menjalankan request CHAT secara serentak (paralel)
    with concurrent.futures.ThreadPoolExecutor(max_workers=JUMLAH_SISWA) as executor:
        hasil_chat = list(executor.map(test_tanya_manual, range(1, JUMLAH_SISWA + 1)))
    
    print("-" * 50)
    print(f"🏁 UJIAN CHAT MANUAL SELESAI! Total Waktu Keseluruhan: {round(time.time() - start_total, 2)} detik")
    print(f"📊 HASIL AKHIR -> Sukses: {hasil_chat.count(True)} | Gagal/Timeout: {hasil_chat.count(False)}")