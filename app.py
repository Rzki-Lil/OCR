import easyocr
import numpy as np
import cv2
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import sys
import time
import traceback

# Inisialisasi aplikasi Flask dengan CORS untuk akses cross-origin
app = Flask(__name__)
CORS(app)  

# Memuat model OCR dengan bahasa Inggris dan Korea
reader = easyocr.Reader(['en', 'ko'], gpu=True)

def merge_nearby_text(results, max_horizontal_gap=50):
    # Fungsi untuk menggabungkan teks yang berdekatan secara horizontal
    # max_horizontal_gap: jarak maksimum antar teks untuk digabungkan

    if not results:
        return []

    # Urutkan hasil berdasarkan posisi x minimum
    sorted_results = sorted(results, key=lambda x: min(point[0] for point in x[0]))
    
    merged = []
    i = 0
    
    while i < len(sorted_results):
        bbox1, text1, conf1 = sorted_results[i]
        min_x1 = min(point[0] for point in bbox1)
        max_x1 = max(point[0] for point in bbox1)
        min_y1 = min(point[1] for point in bbox1)
        max_y1 = max(point[1] for point in bbox1)
        
        merged_with_next = False

        # Cek apakah ada teks berikutnya yang bisa digabungkan
        j = i + 1
        while j < len(sorted_results):
            bbox2, text2, conf2 = sorted_results[j]
            min_x2 = min(point[0] for point in bbox2)
            max_x2 = max(point[0] for point in bbox2)
            min_y2 = min(point[1] for point in bbox2)
            max_y2 = max(point[1] for point in bbox2)

            # Hitung jarak horizontal antara dua kotak teks
            horizontal_gap = min_x2 - max_x1

            # Cek apakah kedua teks berada pada baris yang sama
            same_line = (min_y1 <= max_y2 and max_y1 >= min_y2)

            if 0 <= horizontal_gap <= max_horizontal_gap and same_line:
                # Gabungkan kotak pembatas dari kedua teks
                all_points = bbox1 + bbox2
                new_bbox = [
                    [min(point[0] for point in all_points), min(point[1] for point in all_points)],  
                    [max(point[0] for point in all_points), min(point[1] for point in all_points)],  
                    [max(point[0] for point in all_points), max(point[1] for point in all_points)],  
                    [min(point[0] for point in all_points), max(point[1] for point in all_points)]   
                ]

                # Tambahkan spasi jika jaraknya cukup besar, jika tidak gabungkan langsung
                if horizontal_gap > 20:  
                    new_text = f"{text1} {text2}"
                else:
                    new_text = f"{text1}{text2}"

                # Hitung nilai kepercayaan baru sebagai rata-rata tertimbang
                new_conf = (len(text1) * conf1 + len(text2) * conf2) / (len(text1) + len(text2))

                bbox1 = new_bbox
                text1 = new_text
                conf1 = new_conf
                min_x1 = min(point[0] for point in bbox1)
                max_x1 = max(point[0] for point in bbox1)
                min_y1 = min(point[1] for point in bbox1)
                max_y1 = max(point[1] for point in bbox1)

                # Hapus hasil yang sudah digabungkan dari daftar
                sorted_results.pop(j)
                merged_with_next = True
            else:
                j += 1

        # Tambahkan hasil yang sudah diproses ke daftar akhir
        merged.append((bbox1, text1, conf1))
        i += 1
    
    return merged

@app.route('/ocr', methods=['POST'])
def process_ocr():
    try:
        # Cek apakah request berisi file gambar atau data base64
        file = request.files.get('image')
        if not file:
            # Proses gambar dari data JSON (base64)
            data = request.json
            if not data or 'image' not in data:
                return jsonify({'error': 'Tidak ada gambar yang diberikan'}), 400

            # Ekstrak dan decode data base64
            image_b64 = data['image']
            if 'base64,' in image_b64:
                image_b64 = image_b64.split('base64,')[1]
            
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            # Proses gambar dari file yang diunggah
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print(f"Memproses file: {file.filename}, ukuran: {len(file_bytes) / 1024:.2f} KB")

        # Perbesar ukuran gambar untuk meningkatkan akurasi
        h, w = img.shape[:2]
        img = cv2.resize(img, (w*3, h*3), interpolation=cv2.INTER_NEAREST)

        # Konversi ke grayscale untuk memudahkan pemrosesan
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Terapkan median blur untuk mengurangi noise
        img = cv2.medianBlur(img, 3) 

        # Tentukan area minat (Region of Interest) berdasarkan rasio tinggi
        height, width = img.shape[:2]
        ROI_RATIOS = [0.30, 0.35, 0.78, 0.83]  

        # Ekstrak dua ROI yang berbeda dari gambar - biasanya area dengan nama pemain
        roi1 = img[int(height * ROI_RATIOS[0]):int(height * ROI_RATIOS[1]), 0:width]
        roi2 = img[int(height * ROI_RATIOS[2]):int(height * ROI_RATIOS[3]), 0:width]
        
        # Lakukan OCR pada kedua ROI
        results1 = reader.readtext(roi1)
        results2 = reader.readtext(roi2)
        
        # Cetak hasil deteksi mentah untuk debugging
        print("\nHASIL DETEKSI:")
        print("-"*40)
        print("Hasil ROI 1:")
        for _, text, conf in results1:
            print(f"  - '{text}' (confidence: {conf:.2f})")
        
        print("\nHasil ROI 2:")
        for _, text, conf in results2:
            print(f"  - '{text}' (confidence: {conf:.2f})")

        # Inisialisasi list untuk menyimpan data pemain yang terdeteksi
        all_players = []

        # Gabungkan teks yang berdekatan untuk hasil yang lebih akurat
        merged_results1 = merge_nearby_text(results1)
        merged_results2 = merge_nearby_text(results2)

        # Proses hasil OCR dari ROI pertama
        for bbox, text, conf in merged_results1:
            # Filter hasil dengan kepercayaan rendah
            if conf > 0.2 and len(text) > 1:

                left_x = min(point[0] for point in bbox)
                all_players.append({
                    "name": text, 
                    "confidence": float(conf),
                    "position": left_x, 
                    "roi": "ROI1"
                })

        # Proses hasil OCR dari ROI kedua
        for bbox, text, conf in merged_results2:
            # Filter hasil dengan kepercayaan rendah
            if conf > 0.2 and len(text) > 1:

                left_x = min(point[0] for point in bbox)
                all_players.append({
                    "name": text, 
                    "confidence": float(conf),
                    "position": left_x,  
                    "roi": "ROI2"
                })
                
        # Hapus duplikat nama pemain, pilih yang memiliki nilai kepercayaan tertinggi
        unique_players = {}
        for player in all_players:
            name = player["name"].lower() 
            if name not in unique_players or player["confidence"] > unique_players[name]["confidence"]:
                unique_players[name] = player

        # Kelompokkan pemain berdasarkan ROI dan urutkan berdasarkan posisi
        roi1_players = [p for p in unique_players.values() if p["roi"] == "ROI1"]
        roi2_players = [p for p in unique_players.values() if p["roi"] == "ROI2"]

        roi1_players.sort(key=lambda x: x["position"])
        roi2_players.sort(key=lambda x: x["position"])

        # Gabungkan hasil dari kedua ROI
        sorted_players = roi1_players + roi2_players

        # Tambahkan pemain dari hasil full scan (jika ada)
        full_players = [p for p in unique_players.values() if p["roi"] == "FULL"]
        full_players.sort(key=lambda x: x["position"])
        sorted_players.extend(full_players)

        # Lewati pemain pertama (biasanya user sendiri)
        if len(sorted_players) > 0:
            print("\nPengguna terdeteksi (Pemain 1 - dilewati):")
            print(f"  - '{sorted_players[0]['name']}' (confidence: {sorted_players[0]['confidence']:.2f})")
            sorted_players = sorted_players[1:]

        # Ambil maksimal 7 pemain lainnya
        final_players = []
        for i, player in enumerate(sorted_players):
            if i < 7: 
                final_players.append({
                    "name": player["name"],
                    "confidence": player["confidence"]
                })

        # Cetak nama pemain yang terdeteksi
        print("\nNAMA PEMAIN YANG TERDETEKSI (berurutan):")
        print("-"*40)
        for i, player in enumerate(final_players):
            playerId = i + 2  
            print(f"Pemain {playerId}: '{player['name']}' (confidence: {player['confidence']:.2f})\n")
        

        # Flush output untuk memastikan log tercatat dengan baik
        sys.stdout.flush()

        # response JSON dengan daftar pemain
        response = jsonify({
            'success': True,
            'players': final_players,
            'timestamp': time.time()  
        })
        
        # Atur header untuk mencegah caching
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
        
    except Exception as e:
        # Tangani error dan berikan response yang sesuai
        print(f"\nERROR: {str(e)}")
        traceback_info = traceback.format_exc()
        print(f"Traceback: {traceback_info}")
        sys.stdout.flush()
        
        response = jsonify({'error': str(e), 'timestamp': time.time()})
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        return response, 500

if __name__ == '__main__':
    # Informasi startup aplikasi
    print("\n" + "="*50)
    print("Server OCR Mulai...")
    print("="*50)
    sys.stdout.flush()
    
    # Matikan caching file statis
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
    # Jalankan server Flask
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)