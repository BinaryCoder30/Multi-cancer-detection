import streamlit as st
import requests
import io
from PIL import Image

st.set_page_config(
    page_title="Multi-Cancer Diagnostic AI",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- CSS for dynamic, premium design ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #0f172a, #020617 80%);
        color: #ffffff;
    }

    /* Header styling */
    h1 {
        text-align: center;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        margin-bottom: 0rem;
    }

    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"], section[data-testid="stSidebar"] > div {
        background: rgba(15, 23, 42, 0.7) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* File Uploader styling */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(56, 189, 248, 0.4);
        border-radius: 16px;
        transition: all 0.3s ease;
        padding: 2rem;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(56, 189, 248, 0.05);
        border-color: #38bdf8;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
        transform: translateY(-2px);
    }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        color: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        text-transform: uppercase;
        font-size: 0.95rem;
        width: 100%;
        box-shadow: 0 8px 20px rgba(56, 189, 248, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 15px 25px rgba(56, 189, 248, 0.4);
        color: white;
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* Alerts / Status boxes */
    .stAlert {
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        color: white !important;
    }
    
    .stAlert p {
        color: white !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Image rounding */
    img {
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

# Replace with your actual FastAPI backend URL
BACKEND_URL = "http://localhost:8000/predict/"

st.title("🧬 Multi-Cancer Diagnostic AI")
st.markdown("<p style='text-align: center; color: #a0a0a0; font-size: 1.1rem; margin-bottom: 2rem;'>Upload a histopathological tissue image. The AI will automatically detect the organ and predict the presence and type of cancer.</p>", unsafe_allow_html=True)

st.sidebar.markdown("<h2 style='color: white;'>Instructions</h2>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style="background-color: rgba(255, 255, 255, 0.03); 
                border: 1px solid rgba(255, 255, 255, 0.1); 
                border-radius: 16px; 
                padding: 1rem; 
                backdrop-filter: blur(10px);
                color: white;">
        <div style="color: white; margin-bottom: 0.5rem;">1. Upload a tissue image (JPG/PNG).</div>
        <div style="color: white; margin-bottom: 0.5rem;">2. Click <b>Predict</b> to get AI results.</div>
        <div style="color: white; font-style: italic; font-size: 0.9em; margin-top: 1rem;">*Note: The AI will automatically detect whether the tissue is Colon or Lung and route it to the correct specialized model.*</div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Drop your tissue image here...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("") # spacing
    
    col_btn_1, col_btn_2, col_btn_3 = st.columns([1, 1, 1])
    with col_btn_2:
        predict_button = st.button("Generate Prediction", use_container_width=True)

    if predict_button:
        with st.spinner("Analyzing image features..."):
            # Prepare the file data for sending via POST
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            
            try:
                response = requests.post(BACKEND_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("### Detection Complete!")
                    
                    col_result1, col_result2 = st.columns(2)
                    with col_result1:
                        st.info(f"**Detected Organ:** {result['organ'].capitalize()}")
                    with col_result2:
                        st.success(f"**Prediction:** {result['prediction']}")
                        
                    confidence = result.get('confidence', 0) * 100
                    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
                    st.progress(int(confidence))
                    
                elif response.status_code == 400:
                    result = response.json()
                    st.error(f"🚨 **{result.get('message', 'Bad Request')}**")
                else:
                    st.error(f"Error from server: {response.status_code}")
                    try:
                        st.write(response.json())
                    except:
                        st.write(response.text)
            except requests.exceptions.ConnectionError:
                st.error("🚨 Could not connect to the backend server. Please make sure the FastAPI backend is running on http://localhost:8000.")
