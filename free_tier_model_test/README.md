# Free Tier Model Test Backend

Backend ini khusus buat ngetes respon model free tier dan sengaja dipisah dari `app.py`.

Model yang dipakai sekarang: `gemma-4-31b-it`.

## Fitur

- Endpoint chat biasa: `POST /chat`
- Endpoint gambar + prompt: `POST /image-chat`
- Health check: `GET /health`
- API key bisa pakai: `GEMINI_API_KEY` atau `FREE_TIER_GEMINI_API_KEY`

## Setup

1. Masuk folder ini.
2. Install dependency:

```powershell
pip install -r requirements.txt
```

3. Bikin file `.env`, lalu isi key free tier kamu.

Contoh isi minimal:

```env
FREE_TIER_GEMINI_API_KEY=API_KEY_FREE_TIER_KAMU
# Alternatif: GEMINI_API_KEY=API_KEY_FREE_TIER_KAMU
FREE_TIER_PORT=5002
```

## Jalanin Backend

```powershell
python backend.py
```

Default jalan di `http://127.0.0.1:5002`.

## Pakai Tester

Jalanin semua test (chat + image):

```powershell
python tester.py --mode both
```

Test chat doang:

```powershell
python tester.py --mode chat --text "Halo, tolong jawab singkat"
```

Test image doang:

```powershell
python tester.py --mode image --image "..\Image_Dummy.jpeg" --prompt "Ini gambar apa?"
```

Kalau backend kamu bukan di 5002:

```powershell
python tester.py --base-url "http://127.0.0.1:5003"
```

## Contoh Request Manual

### Chat

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:5002/chat" -Method Post -ContentType "application/json" -Body '{"text":"Halo, kamu siapa?","thinking_level":"HIGH"}'
```

### Image Chat (multipart)

```powershell
curl -X POST "http://127.0.0.1:5002/image-chat" ^
  -F "image_file=@..\Image_Dummy.jpeg" ^
  -F "prompt=Ini gambar apa?" ^
  -F "thinking_level=HIGH"
```
