# Hosting Sistem BI (Apriori + K-Means) - Serafina Paebesi

Login: username & password boleh APA SAJA (asal tidak kosong).

## Isi folder
tugas_8.py          -> aplikasi (sudah dipatch: SECRET_KEY dari environment)
requirements.txt    -> daftar library
Procfile            -> perintah start (Railway / Heroku-style)
render.yaml         -> konfigurasi otomatis Render
static/             -> TARUH logo_unimor.png & foto_serafina.jpg DI SINI
(opsional) P8. UTS_Business Intelligence.xlsx -> agar tombol "1 klik" jalan

## A. Render (link permanen, gratis)
1. Upload semua isi folder ini ke repo GitHub (jangan lupa folder static/ + 2 gambar).
2. render.com -> New -> Blueprint -> pilih repo -> Apply.
   (atau New -> Web Service, lalu isi:
    Build  : pip install -r requirements.txt
    Start  : gunicorn tugas_8:app --workers 1 --threads 4 --timeout 180)
3. Tunggu build selesai -> buka URL https://....onrender.com

Catatan: paket gratis "tidur" kalau lama tidak dibuka (buka pertama agak lambat),
RAM ~512MB. Harus tetap --workers 1 karena data disimpan di memori.

## B. PythonAnywhere (gratis, tidak tidur)
1. Tab Files -> upload tugas_8.py + folder static/ (+ file Excel) ke /home/USERNAME/mysite/
2. Tab Consoles -> Bash:  pip install --user plotly openpyxl
3. Tab Web -> Add a new web app -> Manual configuration (Flask) -> pilih Python 3.10+
4. Buka file WSGI, ganti isinya jadi:

   import sys
   path = "/home/USERNAME/mysite"
   if path not in sys.path:
       sys.path.insert(0, path)
   from tugas_8 import app as application

5. Klik Reload -> buka https://USERNAME.pythonanywhere.com

## C. Demo cepat dari laptop
   python tugas_8.py
   cloudflared tunnel --url http://127.0.0.1:5000     (atau ngrok http 5000)
