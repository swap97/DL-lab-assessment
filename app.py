import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

# --- App Configuration ---
st.set_page_config(
    page_title="Mask Compliance Monitor", 
    page_icon="🛡️",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Smart Surveillance: Face Mask Detector")
st.write("This system uses an optimized **MobileNetV2** model to monitor mask compliance in real-time.")

# --- Load Optimized TFLite Model ---
@st.cache_resource
def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{model_path}' exists.")
        return None

# Use the path where your quantized model is saved
MODEL_PATH = "models/mask_detector_quantized.tflite" 
interpreter = load_tflite_model(MODEL_PATH)

if interpreter:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Determine which output index is for classification and which is for bbox
    # Output 0 is usually class (shape [1, 3]) and Output 1 is bbox (shape [1, 4])
    # However, it's safer to check the last dimension
    if output_details[0]['shape'][-1] == 3:
        cls_idx, box_idx = 0, 1
    else:
        cls_idx, box_idx = 1, 0

    # --- Utility Functions ---
    def predict(image_np):
        # 1. Preprocess: Resize to 224x224 (as defined in IMG_SIZE)
        img_resized = cv2.resize(image_np, (224, 224))
        
        # 2. Normalize and cast to float32
        img_input = np.expand_dims(img_resized, axis=0).astype(np.float32) / 255.0
        
        # 3. Set tensor and Invoke
        interpreter.set_tensor(input_details[0]['index'], img_input)
        interpreter.invoke()
        
        # 4. Extract results
        cls_pred = interpreter.get_tensor(output_details[cls_idx]['index']) 
        box_pred = interpreter.get_tensor(output_details[box_idx]['index']) 
        return cls_pred[0], box_pred[0]

    # --- Main UI ---
    uploaded_file = st.file_uploader("Upload a surveillance image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load Image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        h, w, _ = image_np.shape
        
        # Prediction
        with st.spinner('Analyzing compliance...'):
            cls_probs, box_coords = predict(image_np)
            
        # Results Mapping (Based on your INV_LABELS)
        labels = ['With Mask', 'Without Mask', 'Incorrectly Worn']
        colors = [(0, 255, 0), (255, 0, 0), (255, 165, 0)] # BGR: Green, Red, Orange
        
        class_idx = np.argmax(cls_probs)
        confidence = cls_probs[class_idx]
        
        # Scale bounding box back [xmin, ymin, xmax, ymax]
        xmin, ymin, xmax, ymax = box_coords
        start_pt = (int(xmin * w), int(ymin * h))
        end_pt = (int(xmax * w), int(ymax * h))
        
        # Visualization
        output_img = image_np.copy()
        cv2.rectangle(output_img, start_pt, end_pt, colors[class_idx], 5)
        
        caption = f"{labels[class_idx]}: {confidence*100:.1f}%"
        # Draw background label for readability
        cv2.putText(output_img, caption, (start_pt[0], start_pt[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors[class_idx], 2)
        
        # Display
        st.image(output_img, caption="Processed Surveillance Frame", use_column_width=True)
        
        # Compliance Notification
        if class_idx == 0:
            st.success(f"✅ Access Granted: {caption}")
        elif class_idx == 1:
            st.error(f"⚠️ Security Alert: {caption}")
        else:
            st.warning(f"⚠️ Policy Warning: {caption}")

else:
    st.info("💡 To use this app, first run your notebook to generate 'models/mask_detector_quantized.tflite'.")