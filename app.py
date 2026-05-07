import os
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "1"

import streamlit as st

# --- Global Page Configuration ---
st.set_page_config(
    page_title="Unified AI Decision Platform",
    page_icon="🛡️",
    layout="wide"
)

st.sidebar.title("Navigation")
st.sidebar.markdown("Welcome to the **Unified AI Decision Platform**")

# Define pages
pages = {
    "Home": [
        st.Page(
            "modules/home.py",
            title="Home Dashboard",
            icon="🏠"
        )
    ],
    "Document Intelligence": [
        st.Page(
            "modules/document_intelligence/resume_relevance.py",
            title="Resume Relevance",
            icon="📄"
        ),
        st.Page(
            "modules/document_intelligence/consent_summary.py",
            title="Consent Summary",
            icon="📝"
        )
    ],
    "Assessment Automation": [
        st.Page(
            "modules/assessment_automation/omr_scoring.py",
            title="OMR Scoring",
            icon="🎓"
        )
    ],
    "Predictive Analytics": [
        st.Page(
            "modules/predictive_analytics/hospital_delay.py",
            title="Hospital Delay Predictor",
            icon="🏥"
        )
    ],
    "Trust & Safety": [
        st.Page(
            "modules/trust_safety/deepfake_spread.py",
            title="Deepfake Spread Analysis",
            icon="🛡️"
        )
    ]
}

pg = st.navigation(pages)
pg.run()