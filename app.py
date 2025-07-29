import os
import requests
import streamlit as st
import torch
from torchvision import models, transforms
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from PIL import Image

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1CX0O9r-QcEx9R9Ie-O_3JrFu0ig_wWHo"
MODEL_PATH = "resnet18_otoscopic.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading ResNet18 model...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            st.success("Download complete.")
        else:
            st.error("Failed to download model. Check Google Drive permissions.")

download_model()

# Class labels
class_labels = ['Acute Otitis Media', 'Cerumen Impaction', 'Chronic Otitis Media', 'Myringosclerosis', 'Normal']

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_model = models.resnet18(pretrained=True)
resnet_model.fc = torch.nn.Linear(resnet_model.fc.in_features, len(class_labels))

try:
    resnet_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

resnet_model.eval().to(device)

# Grad-CAM setup
cam_extractor = GradCAM(resnet_model, target_layer="layer4")
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

# Streamlit app
st.title("🩺 Otoscopic Classifier with Grad-CAM (ResNet18)")

uploaded_file = st.file_uploader("📤 Upload an ear image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(device)
    output = resnet_model(input_tensor)
    pred_class = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    st.success(f"Prediction: {class_labels[pred_class]}")
    st.info(f"Confidence: {confidence:.2f}")

    activation_map = cam_extractor(pred_class, output)
    cam_image = overlay_mask(image, activation_map[0].resize(image.size), alpha=0.5)
    st.image(cam_image, caption="Grad-CAM Heatmap", use_column_width=True)
