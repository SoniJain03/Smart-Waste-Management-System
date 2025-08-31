import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import qrcode
import io
from PIL import Image

# ------------------ Sidebar: About ------------------
st.sidebar.title("About")
st.sidebar.image("https://img.icons8.com/dusk/64/recycle-sign.png", width=80)
st.sidebar.markdown("""
**Smart Waste Segregation System**  

An AI-powered web application that automatically classifies waste materials into 12 categories and provides tailored recycling guidance to promote sustainable waste management practices.

**Developer:**  
Soni Jain

**Environmental Impact:**  
This tool helps reduce landfill waste, improve recycling rates, and promote circular economy practices through AI-powered waste management solutions.
""")

# ------------------ App Title ------------------
st.title("♻️ Smart Waste Segregation System")
st.write("Upload an image of waste or use your camera, and the model will predict its class.")

# ------------------ Load Model ------------------
@st.cache_resource
def load_garbage_model():
    try:
        model = load_model("garbage_classification_model.h5")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

model = load_garbage_model()

# Class labels (EXACT order from Colab training)
class_labels = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# Recycling information for each waste category
recycling_info = {
    'battery': {
        'title': '🔋 Battery Recycling',
        'methods': [
            'Take to designated battery recycling drop-off points',
            'Many electronics stores accept batteries for recycling',
            'Never dispose of in regular trash (fire hazard)',
            'Check for local household hazardous waste collection programs'
        ],
        'tips': 'Store used batteries in a non-metal container until you can recycle them.'
    },
    'biological': {
        'title': '🍃 Biological/Organic Waste Recycling',
        'methods': [
            'Compost at home using a compost bin or pile',
            'Use municipal green waste collection services',
            'Create vermicompost (worm composting)',
            'Some communities have anaerobic digestion facilities'
        ],
        'tips': 'Avoid composting meat, dairy, or oily foods to prevent pests.'
    },
    'brown-glass': {
        'title': '🟫 Brown Glass Recycling',
        'methods': [
            'Rinse containers before recycling',
            'Place in glass recycling bin if available',
            'Remove lids and caps (recycle separately)',
            'Take to glass bottle banks if curbside not available'
        ],
        'tips': 'Brown glass is often used for beer bottles and some food containers.'
    },
    'cardboard': {
        'title': '📦 Cardboard Recycling',
        'methods': [
            'Flatten boxes to save space',
            'Place in recycling bin or take to recycling center',
            'Remove any plastic packaging or tape',
            'Keep dry to maintain paper fiber quality'
        ],
        'tips': 'Cardboard can typically be recycled 5-7 times before fibers become too short.'
    },
    'clothes': {
        'title': '👕 Clothing/Textile Recycling',
        'methods': [
            'Donate wearable clothes to charity organizations',
            'Use textile recycling bins for damaged items',
            'Some brands offer clothing take-back programs',
            'Repurpose into cleaning rags or craft materials'
        ],
        'tips': 'Even stained or torn clothing can be recycled—just not in regular trash!'
    },
    'green-glass': {
        'title': '🟩 Green Glass Recycling',
        'methods': [
            'Rinse containers before recycling',
            'Place in glass recycling bin',
            'Separate by color if required by your municipality',
            'Take to glass bottle banks if needed'
        ],
        'tips': 'Green glass is commonly used for wine bottles and some beverage containers.'
    },
    'metal': {
        'title': '🥫 Metal Recycling',
        'methods': [
            'Rinse food cans before recycling',
            'Place in recycling bin or take to scrap metal dealer',
            'Aluminum foil can be recycled if clean',
            'Separate different metals if required'
        ],
        'tips': 'Aluminum can be recycled infinitely without loss of quality!'
    },
    'paper': {
        'title': '📄 Paper Recycling',
        'methods': [
            'Place in paper recycling bin',
            'Remove any plastic windows from envelopes',
            'Keep dry and free from food contamination',
            'Shredded paper should be bagged (check local rules)'
        ],
        'tips': 'Most paper can be recycled 5-7 times before fibers become too short for papermaking.'
    },
    'plastic': {
        'title': '🧴 Plastic Recycling',
        'methods': [
            'Check resin code (number inside triangle)',
            'Rinse containers before recycling',
            'Remove caps and pumps (often different plastic)',
            'Follow local guidelines for which plastics are accepted'
        ],
        'tips': 'Not all plastics are recyclable—check with your local recycling program.'
    },
    'shoes': {
        'title': '👟 Shoe Recycling',
        'methods': [
            'Donate wearable shoes to charity organizations',
            'Some shoe stores have take-back programs',
            'Repurpose old shoes for gardening or sports',
            'Specialized recyclers can separate materials'
        ],
        'tips': 'Paired shoes are much more valuable for donation—keep them together!'
    },
    'trash': {
        'title': '🗑️ Non-Recyclable Waste',
        'methods': [
            'Dispose of in regular trash bin',
            'Minimize creation by choosing products with less packaging',
            'Consider if items can be repaired or repurposed',
            'Follow local guidelines for hazardous waste disposal'
        ],
        'tips': 'When in doubt, check with your local waste management authority.'
    },
    'white-glass': {
        'title': '⚪ White/Clear Glass Recycling',
        'methods': [
            'Rinse containers before recycling',
            'Place in glass recycling bin',
            'Separate by color if required by your municipality',
            'Remove any metal or plastic components'
        ],
        'tips': 'Clear glass has the highest recycling value as it can be used to make new clear glass containers.'
    }
}

# ------------------ Input Selection ------------------
input_method = st.radio(
    "Choose input method:",
    ["Upload an image", "Use camera"],
    horizontal=True
)

# Function to process image and make prediction
def process_and_predict(image):
    # Convert to RGB if needed
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3 and image.shape[2] == 3:  # Already RGB
            img_rgb = image
        else:  # Convert from BGR to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Convert PIL Image to numpy array
        img_rgb = np.array(image.convert('RGB'))
    
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    st.image(img_resized, caption='Input Image', use_container_width=True)
    
    # Preprocess image (EXACTLY like Colab: scale to [0, 1] range)
    img_array = img_resized.astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    try:
        pred = model.predict(img_array, verbose=0)
        class_idx = np.argmax(pred)
        class_name = class_labels[class_idx]
        confidence = np.max(pred)
        
        # Display prediction
        st.subheader("Prediction Results")
        st.success(f"**Predicted Class:** {class_name}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")
        
        # Show recycling information based on prediction
        st.markdown("---")
        st.subheader("♻️ Recycling Guidance")
        
        if class_name in recycling_info:
            info = recycling_info[class_name]
            
            st.markdown(f"### {info['title']}")
            st.markdown("**Proper Recycling Methods:**")
            for method in info['methods']:
                st.markdown(f"- {method}")
            
            st.markdown("**Pro Tip:**")
            st.info(info['tips'])
        else:
            st.warning("Recycling information not available for this category.")
        
        # Show all predictions with progress bars
        with st.expander("View Detailed Predictions"):
            st.write("**All class probabilities:**")
            for i, (label, score) in enumerate(zip(class_labels, pred[0])):
                percentage = score * 100
                st.write(f"{label}: {percentage:.2f}%")
                st.progress(float(score))
                
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# ------------------ Image Upload ------------------
if input_method == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    
    if uploaded_file is not None and model is not None:
        # Load image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img_array is not None:
            process_and_predict(img_array)
        else:
            st.error("Failed to decode the image. Please try another file.")

# ------------------ Camera Capture ------------------
elif input_method == "Use camera":
    st.info("Please allow camera access when prompted by your browser.")
    camera_img = st.camera_input("Take a picture of your waste item")
    
    if camera_img is not None and model is not None:
        # Convert to PIL Image
        image = Image.open(camera_img)
        process_and_predict(image)

# ------------------ Recycling Centers by State & QR Code ------------------
st.markdown("---")
st.subheader("♻️ Find Recycling Centers in Your State")

states = ["Delhi", "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
          "West Bengal", "Gujarat", "Rajasthan", "Punjab", "Haryana"]

selected_state = st.selectbox("Select your state:", states)

if selected_state:
    try:
        # Google Maps search URL
        maps_url = f"https://www.google.com/maps/search/recycling+center+in+{selected_state.replace(' ','+')}"

        # Generate QR Code
        qr = qrcode.QRCode(box_size=10, border=2)
        qr.add_data(maps_url)
        qr.make(fit=True)
        img_qr = qr.make_image(fill_color="black", back_color="white")

        # Convert PIL Image to BytesIO for Streamlit
        buf = io.BytesIO()
        img_qr.save(buf, format="PNG")
        buf.seek(0)

        # Display QR Code
        st.image(buf, caption=f"Scan to see recycling centers in {selected_state}", width=250)
        st.write(f"[🌍 Open Google Maps]({maps_url})")
        
    except Exception as e:
        st.error(f"Error generating QR code: {e}")

# ------------------ General Recycling Tips ------------------
st.markdown("---")
st.subheader("🌱 General Recycling Tips")

general_tips = [
    "♻️ Always rinse containers before recycling to avoid contamination",
    "📋 Check your local recycling guidelines—they vary by location",
    "🔄 When in doubt, find out! Don't wishcycle (putting non-recyclables in recycling)",
    "📦 Flatten cardboard boxes to save space in recycling bins",
    "🏷️ Remove plastic film from packaging before recycling",
    "🔋 Never put batteries or electronics in regular recycling bins",
    "🌧️ Keep recycling dry—wet paper can't be recycled",
    "👕 Consider repair and reuse before recycling"
]

for tip in general_tips:
    st.write(tip)

# ------------------ Footer ------------------
st.markdown("---")
st.caption("Developed by Soni Jain | Smart Waste Segregation System | ♻️ Promoting Sustainable Waste Management")