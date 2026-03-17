import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.disease_predictor import predict_disease, get_symptom_list
from src.clinic_finder import find_nearby_clinics
from src.diet_plan import generate_diet_plan, save_diet_plan_pdf
from src.scraper import search_health_content
from src.credibility import evaluate_results
from src.rag import run_rag_pipeline

st.set_page_config(
    page_title="HealthLens AI",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main {
        background-color: #0a0f1e;
        color: #e8eaf0;
    }

    h1, h2, h3 {
        font-family: 'DM Serif Display', serif !important;
        color: #ffffff !important;
    }

    .hero-title {
        font-family: 'DM Serif Display', serif;
        font-size: 3.5rem;
        color: #ffffff;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #8892a4;
        margin-bottom: 2rem;
    }

    .prediction-card {
        background: linear-gradient(135deg, #1a2035 0%, #1e2847 100%);
        border: 1px solid #2a3558;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .disease-name {
        font-family: 'DM Serif Display', serif;
        font-size: 1.8rem;
        color: #60a5fa;
        margin-bottom: 0.5rem;
    }

    .confidence-high { color: #34d399; font-weight: 600; }
    .confidence-med  { color: #fbbf24; font-weight: 600; }
    .confidence-low  { color: #f87171; font-weight: 600; }

    .clinic-card {
        background: #131929;
        border: 1px solid #1e2d4a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .section-label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #4a6080;
        margin-bottom: 0.5rem;
    }

    .tag {
        display: inline-block;
        background: #1e3a5f;
        color: #93c5fd;
        border-radius: 20px;
        padding: 0.2rem 0.8rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }

    .stMultiSelect > div {
        background: #131929 !important;
        border-color: #2a3558 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-family: 'DM Sans', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af);
        transform: translateY(-1px);
    }

    div[data-testid="stMetricValue"] {
        font-family: 'DM Serif Display', serif;
        color: #60a5fa !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="section-label">AI-Powered Health Intelligence</p>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">HealthLens AI🔬</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Select your symptoms · Get instant disease prediction · Find verified information & nearby doctors</p>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    city = st.text_input("📍 Your City", value="Mumbai", help="Used to find nearby clinics")
    st.markdown("---")
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. **Select symptoms** from dropdown
    2. **AI predicts** disease using XGBoost + Neural Network ensemble
    3. **RAG pipeline** fetches verified medical info
    4. **Find clinics** near your city
    5. **Download** personalized diet plan PDF
    """)
    st.markdown("---")
    st.markdown("### 🏥 Supported Diseases")
    st.caption("This tool can predict the following 41 conditions:")
    
    diseases = sorted([
        "Fungal infection", "Allergy", "GERD", "Chronic cholestasis",
        "Drug Reaction", "Peptic ulcer disease", "AIDS", "Diabetes",
        "Gastroenteritis", "Bronchial Asthma", "Hypertension", "Migraine",
        "Cervical spondylosis", "Paralysis (brain hemorrhage)", "Jaundice",
        "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
        "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E",
        "Alcoholic hepatitis", "Tuberculosis", "Common Cold", "Pneumonia",
        "Dimorphic hemmorhoids(piles)", "Heart attack", "Varicose veins",
        "Hypothyroidism", "Hyperthyroidism", "Hypoglycemia", "Osteoarthritis",
        "Arthritis", "(vertigo) Paroymsal Positional Vertigo", "Acne",
        "Urinary tract infection", "Psoriasis", "Impetigo"
    ])
    
    for disease in diseases:
        st.caption(f"• {disease}")
    st.markdown("---")
    st.caption("⚠️ This tool covers 41 common diseases and uses a synthetic dataset. Results are indicative only. Always consult a qualified doctor.")


col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🩺 Select Your Symptoms")
    st.markdown('<p class="section-label">Choose all symptoms that apply</p>', unsafe_allow_html=True)

    all_symptoms = get_symptom_list()
    selected_symptoms = st.multiselect(
        label="Symptoms",
        options=all_symptoms,
        placeholder="Type or select symptoms...",
        label_visibility="collapsed"
    )

    if selected_symptoms:
        st.markdown("**Selected:**")
        tags_html = "".join([f'<span class="tag">{s}</span>' for s in selected_symptoms])
        st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔬 Analyze Symptoms", use_container_width=True)

with col2:
    if analyze_btn and selected_symptoms:
        with st.spinner("Running ML models..."):
            result = predict_disease(selected_symptoms)

        if "error" in result:
            st.error(result["error"])
        else:
            predictions = result["ensemble_predictions"]
            top = predictions[0]

            # Top prediction
            conf = top["confidence"]
            conf_class = "confidence-high" if conf >= 70 else "confidence-med" if conf >= 40 else "confidence-low"

            st.markdown(f"""
            <div class="prediction-card">
                <p class="section-label">Top Prediction</p>
                <p class="disease-name">{top["disease"]}</p>
                <p class="{conf_class}">Confidence: {conf}%</p>
            </div>
            """, unsafe_allow_html=True)

            # Model comparison
            st.markdown("**Model Breakdown**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("XGBoost", f"{result['xgboost_predictions'][0]['confidence']}%")
            with c2:
                st.metric("Neural Net", f"{result['neural_net_predictions'][0]['confidence']}%")
            with c3:
                st.metric("Ensemble", f"{conf}%")

            # Other predictions
            st.markdown("**Other Possibilities**")
            for p in predictions[1:]:
                st.progress(int(p["confidence"]), text=f"{p['disease']} — {p['confidence']}%")

            # Store top disease in session
            st.session_state["top_disease"] = top["disease"]
            st.session_state["symptoms"] = selected_symptoms

    elif analyze_btn and not selected_symptoms:
        st.warning("Please select at least one symptom!")

# ---- Bottom Sections ----
if "top_disease" in st.session_state:
    disease = st.session_state["top_disease"]
    symptoms = st.session_state["symptoms"]

    st.divider()
    tab1, tab2, tab3 = st.tabs(["📋 Medical Summary", "🏥 Nearby Clinics", "🥗 Diet Plan"])

    # ---- Tab 1: RAG Summary ----
    with tab1:
        with st.spinner("Fetching verified medical information..."):
            web_results = search_health_content(disease)
            evaluated = evaluate_results(web_results)
            credible = [r for r in evaluated if r["credibility_score"] >= 0.25]
            rag_output = run_rag_pipeline(disease, credible)

        st.markdown(f"### About {disease}")
        st.markdown(rag_output["summary"])

        if rag_output.get("sources"):
            st.markdown("**Verified Sources:**")
            for s in rag_output["sources"]:
                st.markdown(f"- [{s}]({s})")

    # ---- Tab 2: Clinic Finder ----
    with tab2:
        with st.spinner("Finding nearby clinics..."):
            clinic_data = find_nearby_clinics(disease, city)

        st.markdown(f"### {clinic_data['specialty'].title()}s near {city}")
        for c in clinic_data["clinics"]:
            st.markdown(f"""
            <div class="clinic-card">
                <strong style="color:#60a5fa">{c['name']}</strong><br>
                📍 {c['address']}<br>
                📞 {c['phone']}<br>
                <a href="{c['maps_link']}" target="_blank" style="color:#34d399">📌 Open in Google Maps</a>
            </div>
            """, unsafe_allow_html=True)

    # ---- Tab 3: Diet Plan ----
    with tab3:
        with st.spinner("Generating personalized diet plan..."):
            diet_text = generate_diet_plan(disease, symptoms)
            pdf_path = save_diet_plan_pdf(disease, diet_text)

        st.markdown(f"### Diet Plan for {disease}")
        st.markdown(diet_text)

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📥 Download Diet Plan PDF",
                data=f,
                file_name=f"HealthLens_DietPlan_{disease.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )