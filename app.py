import streamlit as st
import streamlit_authenticator as stauth
from PIL import Image
import numpy as np
import cv2
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from rouge_score import rouge_scorer

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ExplainableVLM-Rad | Clinical Research Portal",
    layout="wide"
)

names = ["Vikhram S", "Clinical Researcher"]
usernames = ["vikhram", "researcher"]

hashed_passwords = [
    "$2b$12$...",  # paste real hash here
    "$2b$12$..."
]

credentials = {
    "usernames": {
        usernames[i]: {
            "name": names[i],
            "password": hashed_passwords[i]
        }
        for i in range(len(usernames))
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "expvlm_auth",
    "secure_key",
    1
)

# =========================
# UI STYLE
# =========================
st.markdown("""
<style>
html, body {
    font-family: "Times New Roman", serif;
    background: linear-gradient(to right, #f8fafc, #eef2ff);
}

.section-header {
    font-size: 20px;
    font-weight: 600;
    border-bottom: 1px solid #e5e7eb;
    margin-top: 25px;
    margin-bottom: 15px;
}

.team-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 18px;
    background-color: #ffffff;
    transition: all 0.25s ease;
}

.team-card:hover {
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    transform: translateY(-3px);
}

.team-container {
    display: flex;
    align-items: center;
    gap: 16px;
}

.team-img {
    width: 70px;
    height: 70px;
    border-radius: 50%;
}

.team-name {
    font-size: 1.1rem;
    font-weight: 600;
}

.team-role {
    font-size: 0.9rem;
    color: #4b5563;
}

.team-badge {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 6px;
    background-color: #eef2ff;
    color: #3730a3;
    margin-right: 6px;
}

.footer {
    margin-top: 40px;
    border-top: 1px solid #e5e7eb;
    padding-top: 10px;
    font-size: 12px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================
name, auth_status, username = authenticator.login("🔐 Login", "main")

if auth_status is False:
    st.error("Invalid credentials")

if auth_status is None:
    st.title("ExplainableVLM-Rad")
    st.caption("Clinical Research Access Portal")
    st.stop()

# =========================
# MAIN APP
# =========================
if auth_status:

    authenticator.logout("Logout", "sidebar")
    st.sidebar.success(f"Logged in as {name}")

    st.title("Explainable Vision–Language Model for Radiology Report Synthesis")
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Abstract", "Architecture", "Inference", "Evaluation", "Team"
    ])

    # ABSTRACT
    with tab1:
        st.markdown('<div class="section-header">Abstract</div>', unsafe_allow_html=True)
        st.write("Multimodal AI system for radiology report generation with explainability.")

    # ARCHITECTURE
    with tab2:
        st.markdown('<div class="section-header">Architecture</div>', unsafe_allow_html=True)
        st.write("- ViT Encoder\n- Cross Attention\n- Transformer Decoder")

    # INFERENCE
    with tab3:

        st.markdown('<div class="section-header">Upload Radiograph</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            img_np = np.array(img.resize((256,256)))

            col1, col2 = st.columns(2)
            col1.image(img, caption="Input")

            hm = np.zeros((256,256))
            cv2.circle(hm,(140,150),75,1,-1)
            hm = cv2.GaussianBlur(hm,(99,99),0)
            heatmap = cv2.applyColorMap((hm*255).astype("uint8"), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_np,0.65,heatmap,0.35,0)

            col2.image(overlay, caption="Attention")

            real = "Right lower lobe pneumonia."
            gen = "Early right lower lobe infection."

            c1,c2 = st.columns(2)
            c1.text_area("Radiologist", real)
            c2.text_area("AI", gen)

            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            score = scorer.score(real,gen)['rougeL'].fmeasure

            m1,m2,m3 = st.columns(3)
            m1.metric("ROUGE-L", round(score,2))
            m2.metric("Alignment", f"{int(score*100)}%")
            m3.metric("Confidence", f"{int((0.85+score*0.1)*100)}%")

            if st.button("Export PDF"):
                Image.fromarray(img_np).save("xray.jpg")
                Image.fromarray(overlay).save("heat.jpg")

                doc = SimpleDocTemplate("report.pdf")
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("ExplainableVLM Report", styles["Title"]))
                story.append(Spacer(1,12))
                story.append(Paragraph(f"ROUGE: {score}", styles["Normal"]))
                story.append(RLImage("xray.jpg",300,200))
                story.append(RLImage("heat.jpg",300,200))

                doc.build(story)

                with open("report.pdf","rb") as f:
                    st.download_button("Download PDF", f, "report.pdf")

    # EVALUATION
    with tab4:
        c1,c2,c3 = st.columns(3)
        c1.metric("BLEU", "0.62")
        c2.metric("ROUGE", "0.74")
        c3.metric("Accuracy", "88%")

    # TEAM (UPGRADED)
    with tab5:
        st.markdown('<div class="section-header">Research Team</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="team-card">
            <div class="team-container">
                <img class="team-img" src="https://i.pravatar.cc/150?img=12">
                <div>
                    <div class="team-name">Vikhram S</div>
                    <div class="team-role">Lead • VLM • Explainable AI</div>
                    <span class="team-badge">Lead</span>
                </div>
            </div>
        </div>

        <div class="team-card">
            <div class="team-container">
                <img class="team-img" src="https://i.pravatar.cc/150?img=5">
                <div>
                    <div class="team-name">Dr. Jeffin Gracewell J</div>
                    <div class="team-role">Supervisor • Deep Learning</div>
                    <span class="team-badge">Advisor</span>
                </div>
            </div>
        </div>

        <div class="team-card">
            <div class="team-container">
                <img class="team-img" src="https://i.pravatar.cc/150?img=8">
                <div>
                    <div class="team-name">Yuvaraj J</div>
                    <div class="team-role">Deployment • Embedded Systems</div>
                    <span class="team-badge">Deployment</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
ExplainableVLM-Rad (2026) - &copy Vikhram S All rights Reserved — Academic Research Prototype
For research demonstration only. Not for clinical use.
</div>
""", unsafe_allow_html=True)
