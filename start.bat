
@echo off
REM Buka CMD pertama untuk Flask
start "" /min cmd /c "python app.py"
timeout /t 20 > nul
REM Buka CMD kedua untuk Ngrok
start "Ngrok Tunnel" /min cmd /k "ngrok http --domain=settled-modern-stinkbug.ngrok-free.app 5000"
