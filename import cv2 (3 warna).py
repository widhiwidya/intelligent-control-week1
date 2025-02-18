import cv2
import numpy as np

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rentang warna dalam HSV
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([85, 255, 255])
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])

    # Masking untuk deteksi warna
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Temukan kontur untuk setiap warna
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def draw_bounding_box(contours, color, label, frame):
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter area untuk menghindari noise
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                accuracy = min(area / (w * h), 1.0) * 100  # Estimasi akurasi
                text = f"{label} ({accuracy:.1f}%)"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Gambar bounding box dan keterangan warna
    draw_bounding_box(contours_red, (0, 0, 255), "Merah", frame)
    draw_bounding_box(contours_green, (0, 255, 0), "Hijau", frame)
    draw_bounding_box(contours_blue, (255, 0, 0), "Biru", frame)

    # Tampilkan hasil
    cv2.imshow("Frame with Bounding Boxes", frame)

    # Keluar jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()