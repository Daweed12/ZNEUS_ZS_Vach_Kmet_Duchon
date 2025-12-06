import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# --- 1. SETUP CONSTANTS (MATCHING YOUR TRAINED MODEL) ---
IMG_SIZE = 128
STD=0.2469
MEAN=0.4704
CLASSES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space']

# Load Model
model = tf.keras.models.load_model(r"C:\Users\alexd\ZNEUS\ZNEUS_ZS_Vach_Kmet_Duchon\live_alex\best_cnn_model.keras")

# --- 2. SETUP MEDIAPIPE ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# static_image_mode=False is for video (faster), max_num_hands=1 simplifies logic
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

def preprocess_for_model(hand_crop):
    """
    Exact preprocessing steps from your notebook:
    Grayscale -> Resize (128x128) -> /255 -> (img - MEAN)/STD
    """
    # Resize to model input size
    img = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
    
    # Convert to Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize 0-1
    img = img.astype('float32') / 255.0
    
    # Standardize using your notebook values
    img = (img - MEAN) / STD
    
    # Reshape: (1, 128, 128, 1)
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    return img

def get_square_bbox(h, w, landmarks, padding=40):
    """
    Calculates a square bounding box around the hand landmarks.
    Square crops are better because resizing a rectangle to a square (128x128) distorts the image.
    """
    x_min = w
    y_min = h
    x_max = 0
    y_max = 0

    # Find boundaries of the hand
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        if x < x_min: x_min = x
        if x > x_max: x_max = x
        if y < y_min: y_min = y
        if y > y_max: y_max = y

    # Add padding
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(w, x_max + padding)
    y_max = min(h, y_max + padding)

    # Make it a Square (take the larger dimension)
    box_w = x_max - x_min
    box_h = y_max - y_min
    max_side = max(box_w, box_h)
    
    # Recalculate center to keep square centered
    center_x = x_min + box_w // 2
    center_y = y_min + box_h // 2
    
    # New square coordinates
    x1 = max(0, center_x - max_side // 2)
    y1 = max(0, center_y - max_side // 2)
    x2 = min(w, x1 + max_side)
    y2 = min(h, y1 + max_side)
    
    return x1, y1, x2, y2

# --- 3. MAIN LOOP ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Mirror frame (natural feel)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # MediaPipe works with RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 1. Draw Skeleton (Optional, looks cool)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 2. Calculate Bounding Box
            x1, y1, x2, y2 = get_square_bbox(h, w, hand_landmarks.landmark)
            
            # 3. Crop Hand
            hand_crop = frame[y1:y2, x1:x2]
            
            # Safety check: ensure crop isn't empty
            if hand_crop.size > 0:
                try:
                    # 4. Preprocess & Predict
                    input_data = preprocess_for_model(hand_crop)
                    prediction = model.predict(input_data, verbose=0)
                    
                    class_idx = np.argmax(prediction)
                    confidence = np.max(prediction)
                    label = CLASSES[class_idx]
                    
                    # 5. Visuals
                    # Draw box around hand
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (x1, y1-30), (x1+200, y1), (0,255,0), -1)
                    cv2.putText(frame, f"{label} ({confidence:.0%})", (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                                
                except Exception as e:
                    print(f"Prediction Error: {e}")

    cv2.imshow('ASL MediaPipe', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()