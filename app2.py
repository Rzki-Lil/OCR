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
from ultralytics import YOLO
import os

# Inisialisasi aplikasi Flask dengan CORS untuk akses cross-origin
app = Flask(__name__)
CORS(app)  

model_path = 'player_card_model.pt'

if not os.path.exists(model_path):
    print(f"Error: Model file '{model_path}' tidak ditemukan")
    sys.exit(1)

print(f"Loading model Yolo {model_path}...")
model = YOLO(model_path)

reader = easyocr.Reader(['en', 'ko'], gpu=True)


@app.route('/ocr', methods=['POST'])
def process_ocr():
    try:
        MAX_FILE_SIZE = 1 * 1024 * 1024  # 1MB in bytes
        
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
            

            image_size = len(base64.b64decode(image_b64))
            if image_size > MAX_FILE_SIZE:
                return jsonify({
                    'error': f'Ukuran gambar ({image_size/1024/1024:.2f}MB) melebihi batas maksimum (1MB)'
                }), 413
            
            image_data = base64.b64decode(image_b64)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            file.seek(0, 2) 
            file_size = file.tell()
            file.seek(0) 
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({
                    'error': f'Ukuran file ({file_size/1024/1024:.2f}MB) melebihi batas maksimum (1MB)'
                }), 413
            
            # Proses gambar dari file yang diunggah
            file_bytes = file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            print(f"Memproses file: {file.filename}, ukuran: {file_size/1024:.2f} KB")

        # Jalankan deteksi dengan YOLO
        print("Menjalankan deteksi YOLO...")
        results = model.predict(source=img, save=False, conf=0.1, device=0) 
        
        # Ambil semua bounding boxes
        all_boxes = []
        for r in results:
            boxes = r.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                all_boxes.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'conf': conf, 'cls': cls,
                    'center_x': (x1 + x2) // 2,
                    'center_y': (y1 + y2) // 2
                })
        
        # Kelompokkan kotak berdasarkan baris
        if all_boxes:
            heights = [box['y2'] - box['y1'] for box in all_boxes]
            avg_height = sum(heights) / len(heights)
            row_threshold = avg_height * 0.7  
            
            # Urutkan berdasarkan y terlebih dahulu untuk mengelompokkan dalam baris
            all_boxes.sort(key=lambda box: box['center_y'])
            
            # Kelompokkan kotak berdasarkan baris
            rows = []
            current_row = [all_boxes[0]]
            
            for box in all_boxes[1:]:
                if abs(box['center_y'] - current_row[0]['center_y']) < row_threshold:
                    current_row.append(box)
                else:
                    current_row.sort(key=lambda box: box['center_x'])
                    rows.append(current_row)
                    current_row = [box]

            if current_row:
                current_row.sort(key=lambda box: box['center_x'])
                rows.append(current_row)
            # Ratakan baris yang telah diurutkan
            sorted_boxes = []
            for row in rows:
                sorted_boxes.extend(row)
            
            # Proses kotak sesuai urutan
            all_players = []
            for i, box in enumerate(sorted_boxes):
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                
                # Hanya gunakan setengah bagian bawah untuk OCR
                # Hitung koordinat tengah secara vertikal
                mid_y = y1 + (y2 - y1) // 2

                bottom_half = img[mid_y:y2, x1:x2]

                scaled_height, scaled_width = bottom_half.shape[:2]
                scale_factor = 2
                new_height, new_width = scaled_height * scale_factor, scaled_width * scale_factor
                card_img = cv2.resize(bottom_half, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

                card_img = cv2.medianBlur(card_img, 3)  
                
                ocr_results = reader.readtext(card_img, detail=1, paragraph=False)

                if ocr_results:
                    # Ambil teks pertama sebagai nama pemain
                    player_name = ocr_results[0][1]
                    player_conf = float(ocr_results[0][2])
                    
                    all_players.append({
                        "name": player_name,
                        "confidence": player_conf,
                        "position": i,

                    })
                     
            # Hapus duplikat nama pemain, pilih yang memiliki nilai kepercayaan tertinggi
            unique_players = {}
            for player in all_players:
                name = player["name"].lower() 
                if name not in unique_players or player["confidence"] > unique_players[name]["confidence"]:
                    unique_players[name] = player
            
            # Urutkan pemain berdasarkan posisi
            sorted_players = sorted(unique_players.values(), key=lambda x: x["position"])
            
            # Lewati pemain pertama (biasanya user sendiri)
            if len(sorted_players) > 0:
                print("\nPengguna terdeteksi (Pemain 1 - dilewati):")
                print(f"  - '{sorted_players[0]['name']}' (confidence: {sorted_players[0]['confidence']:.2f})")
                sorted_players = sorted_players[1:]

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
                print(f"Pemain {playerId}: '{player['name']}' (confidence: {player['confidence']:.2f})")
            
        else:
            print("Tidak ada player card terdeteksi oleh YOLO")
            final_players = []
        
        # Flush output untuk memastikan log tercatat dengan baik
        sys.stdout.flush()

        # response JSON dengan daftar pemain
        response_data = {
            'success': True,
            'players': final_players,
            'timestamp': time.time()
        }
        
        response = jsonify(response_data)

        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        return response
        
    except Exception as e:
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
