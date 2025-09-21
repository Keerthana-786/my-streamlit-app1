import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import tempfile

# Allowed image extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_image(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS

# Preprocess image
def preprocess_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Enhance contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Sharpen
    blur = cv2.GaussianBlur(img, (0,0), 3)
    img = cv2.addWeighted(img, 1.5, blur, -0.5, 0)

    # Deskew
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return img

# Detect filled bubbles
def detect_bubbles(image):
    img = preprocess_image(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    contours_data = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

    detected = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 1500:
            (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
            circle_area = np.pi * (radius**2)
            if area / circle_area < 0.7:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            roi = thresh[y:y+h, x:x+w]
            fill_ratio = cv2.countNonZero(roi) / (w*h)
            if fill_ratio > 0.5:
                detected.append((int(x_c), int(y_c)))
                mask = np.zeros_like(roi)
                mask[roi>0] = 255
                colored_roi = img[y:y+h, x:x+w]
                colored_roi[mask==255] = [0, 255, 0]
                img[y:y+h, x:x+w] = colored_roi
    return detected, img

# Map bubbles to options
def map_bubbles(detected):
    detected_sorted = sorted(detected, key=lambda b: (b[1], b[0]))
    answers = []
    for idx, bubble in enumerate(detected_sorted):
        answers.append(chr(65 + (idx % 4)))  # A-D cycle
    return answers

# --- Streamlit UI ---
st.title("📄 OMR Sheet Grader")
st.write("Upload your OMR sheet and answer key to get the results.")

omr_file = st.file_uploader("Upload OMR sheet (PNG/JPG)", type=list(ALLOWED_IMAGE_EXTENSIONS))
answer_key_file = st.file_uploader("Upload Answer Key (TXT/CSV/XLSX)")

if st.button("Process"):

    if omr_file is None or answer_key_file is None:
        st.error("Please upload both OMR sheet and answer key.")
    else:
        if not allowed_image(omr_file.name):
            st.error("OMR sheet must be an image (PNG, JPG, JPEG).")
        else:
            # Read OMR sheet
            image = Image.open(omr_file)

            # Detect bubbles
            detected_omr, processed_omr = detect_bubbles(image)
            if len(detected_omr) == 0:
                st.warning("No bubbles detected on OMR sheet.")
            else:
                omr_answers = map_bubbles(detected_omr)

                # Read answer key
                try:
                    if answer_key_file.name.lower().endswith(('.xls', '.xlsx')):
                        df = pd.read_excel(answer_key_file)
                        key_answers = [str(a).upper() for a in df['Answer'].tolist()]
                    else:
                        key_answers = [line.strip().upper() for line in answer_key_file.getvalue().decode().splitlines()]
                except:
                    key_answers = ['A'] * len(omr_answers)

                total_questions = min(len(omr_answers), len(key_answers))
                correct_count = sum([1 for i in range(total_questions) if omr_answers[i]==key_answers[i]])
                score = 0 if total_questions == 0 else round((correct_count/total_questions)*100, 2)

                # Feedback table
                feedback = []
                for i in range(total_questions):
                    status = "Correct" if omr_answers[i]==key_answers[i] else "Wrong"
                    feedback.append((i+1, status, key_answers[i]))

                st.subheader("✅ Results")
                st.write(f"Detected Bubbles: {len(omr_answers)}")
                st.write(f"Total Questions: {total_questions}")
                st.write(f"Score: {score}%")

                st.subheader("Feedback per question")
                st.dataframe(pd.DataFrame(feedback, columns=["Q#", "Status", "Answer"]))

                st.subheader("Processed OMR Sheet")
                st.image(cv2.cvtColor(processed_omr, cv2.COLOR_BGR2RGB), use_column_width=True)
