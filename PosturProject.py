import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

# MediaPipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Kamera ve Tkinter Ayarları
cap = cv2.VideoCapture(0)
root = tk.Tk()
root.title("Posture Correction Mascot")
root.geometry("800x600")

# Kalibrasyon ve duruş izleme değişkenleri
is_calibrated = False
shoulder_threshold = neck_threshold = 0
calibration_shoulder_angles = []
calibration_neck_angles = []
calibration_frames = 0

# Burun yüksekliği için sabit threshold
nose_threshold = 245  # Burun pozisyonu 245 pikselin üstüne çıkarsa kötü duruş olarak algılanacak

# Maskot görüntü
mascot_img = Image.open("C:/Users/Volkan/Desktop/PosturDetect/mascotn.png").resize((150, 150), Image.LANCZOS)
mascot_photo = ImageTk.PhotoImage(mascot_img)

# Kamera akışını gösteren Tkinter kodu
camera_label = tk.Label(root)
camera_label.pack()

# Kötü duruş kontrolü için değişkenler
mascot_window = None
mascot_displayed = False
good_posture_time = 3
Bad_posture_time= 5

"""noktalar arası açı hesaplama en az 3 nokta gerekli"""
def calculate_angle(a, b, c): 
    ab = np.array(b) - np.array(a)
    bc = np.array(c) - np.array(b)
    angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ab[1], ab[0])
    return abs(angle * 180.0 / np.pi)

def show_mascot():
    """Maskotun masaüstünde kalıcı olarak görünmesini sağlar."""
    global mascot_window, mascot_displayed

    # Eğer zaten bir mascot penceresi varsa, tekrar oluşturma
    if mascot_window and mascot_window.winfo_exists():
        return

    mascot_displayed = True

    # Yeni bir Tkinter penceresi oluşturur ve masaüstü üzerinde gösterir
    mascot_window = tk.Toplevel()
    mascot_window.overrideredirect(True)  
    mascot_window.wm_attributes("-topmost", True)  
    mascot_window.wm_attributes("-transparentcolor", "white")  # Arka planı şeffaf yapar

    # Pencere boyutları
    screen_width = mascot_window.winfo_screenwidth()
    screen_height = mascot_window.winfo_screenheight()

    # Maskotu pencereye ekler
    mascot_label = tk.Label(mascot_window, image=mascot_photo, bg="white")
    mascot_label.pack()

    # Maskot başlangıç pozisyonu 
    mascot_x_position = screen_width // 2 - 75  # Ekranın ortasında yatayda ortalanmış
    mascot_y_position = screen_height - 200     # Alt kısımda gösterilecek

    mascot_window.geometry(f"+{mascot_x_position}+{mascot_y_position}")

def close_mascot():
    """Maskot penceresini kapatır."""
    global mascot_window, mascot_displayed, good_posture_time
    if mascot_window and mascot_window.winfo_exists():
        mascot_window.destroy()
    mascot_displayed = False
    good_posture_time = None

def update_frame():
    global is_calibrated, calibration_frames, shoulder_threshold, neck_threshold
    global mascot_displayed, good_posture_time

    # Kameradan görüntü alma
    ret, frame = cap.read()
    if not ret:
        root.quit()  #Kameradan görüntü alınamıyorsa çıkış yap
        return

    #Görüntü işleme
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Poz verisi varsa işlem yapar
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Anahtar noktaları aldığımız kısım
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))
        nose = (int(landmarks[mp_pose.PoseLandmark.NOSE.value].x * frame.shape[1]),
                int(landmarks[mp_pose.PoseLandmark.NOSE.value].y * frame.shape[0]))

        # Omuzların yatay hattı ve kafa açısı hesaplama
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50))
        head_tilt_angle = calculate_angle(left_ear, right_ear, right_shoulder)

        # Kalibrasyon
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(head_tilt_angle)
            calibration_frames += 1
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 5
            neck_threshold = np.mean(calibration_neck_angles) - 3
            is_calibrated = True

        # Duruş kontrolü
        if is_calibrated:
            # Kötü duruş tespiti
            if shoulder_angle < shoulder_threshold or head_tilt_angle < neck_threshold or nose[1] > nose_threshold:
                # Eğer maskot zaten gösterilmediyse, göster
                if not mascot_displayed:
                    show_mascot()
                    good_posture_time = None
            else:
                # İyi duruş yakalandığında
                if not good_posture_time:
                    good_posture_time = time.time()
                
                # Kullanıcı belirli bir süre iyi duruş sergiler ise maskotu kapat
                if good_posture_time and time.time() - good_posture_time >= Bad_posture_time:
                    close_mascot()

        # Anahtar noktaları ve çizgileri gösterme
        cv2.circle(frame, left_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, right_shoulder, 5, (0, 255, 0), -1)
        cv2.circle(frame, left_ear, 5, (255, 0, 0), -1)
        cv2.circle(frame, right_ear, 5, (255, 0, 0), -1)
        cv2.circle(frame, nose, 5, (0, 0, 255), -1)

        # Çizgileri çiz
        cv2.line(frame, left_shoulder, right_shoulder, (0, 255, 0), 2)
        cv2.line(frame, left_ear, right_ear, (255, 0, 0), 2)
        cv2.line(frame, right_shoulder, (right_shoulder[0], right_shoulder[1] - 50), (0, 255, 255), 2)

        # Açıyı kamerada gösterme
        cv2.putText(frame, f'Shoulder Angle: {int(shoulder_angle)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f'Head Tilt Angle: {int(head_tilt_angle)}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f'Nose Y Position: {nose[1]}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # OpenCV görüntüsünü Tkinter uyumlu hale getirir
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    camera_label.imgtk = imgtk
    camera_label.configure(image=imgtk)

    root.after(10, update_frame)

# İlk çerçeveyi yükle ve döngüyü başlat
root.after(10, update_frame)
root.mainloop()

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()