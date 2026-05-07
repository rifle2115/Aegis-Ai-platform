import streamlit as st
import pdfplumber
import docx2txt
import io
import spacy
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

@st.cache_resource
def load_spacy_model():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_model():
    # Switched to a more powerful, all-around model for better semantic similarity
    return SentenceTransformer('all-mpnet-base-v2')

nlp = load_spacy_model()
sentence_model = load_sentence_model()

def clean_text(text):
    """Cleans text by removing extra whitespace and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_text(uploaded_file):
    if not uploaded_file: return ""
    try:
        file_obj = io.BytesIO(uploaded_file.getvalue())
        if uploaded_file.type == "application/pdf":
            with pdfplumber.open(file_obj) as pdf:
                return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        else:
            return docx2txt.process(file_obj)
    except Exception as e:
        st.error(f"Error reading {uploaded_file.name}: {e}")
        return ""

SKILLS_DB = [
    'python', 'java', 'c++', 'sql', 'mysql', 'postgresql', 'mongodb', 'nosql',
    'javascript', 'typescript', 'react', 'angular', 'vue', 'nodejs', 'expressjs',
    'html', 'css', 'tailwind', 'bootstrap', 'git', 'github', 'docker', 'kubernetes',
    'aws', 'azure', 'gcp', 'jira', 'confluence', 'agile', 'scrum', 'machine learning',
    'deep learning', 'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'data analysis',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'natural language processing', 'nlp',
    'spacy', 'nltk', 'power bi', 'tableau', 'excel', 'data visualization',
    'communication', 'teamwork', 'problem solving', 'leadership'
]

def extract_skills(text):
    if not text: return []
    doc = nlp(text.lower())
    found_skills = {skill for skill in SKILLS_DB if re.search(r'\b' + re.escape(skill) + r'\b', text.lower())}
    found_skills.update(chunk.text.lower().strip() for chunk in doc.noun_chunks if chunk.text.lower().strip() in SKILLS_DB)
    return sorted(list(found_skills))

def get_verdict(score):
    if score >= 90: return "Excellent Fit"
    elif score >= 60: return "High Fit"
    elif score >= 40: return "Medium Fit"
    else: return "Low Fit"

def analyze_documents(jd_text, resume_text):
    jd_skills, resume_skills = extract_skills(jd_text), extract_skills(resume_text)
    matching_skills = set(jd_skills) & set(resume_skills)
    missing_skills = set(jd_skills) - set(resume_skills)
    hard_match_score = (len(matching_skills) / len(jd_skills)) * 100 if jd_skills else 0
    
    semantic_score = 0
    if jd_text and resume_text:
        # Clean the text before encoding for better accuracy
        jd_cleaned = clean_text(jd_text)
        resume_cleaned = clean_text(resume_text)
        embeddings = sentence_model.encode([jd_cleaned, resume_cleaned])
        semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        semantic_score = max(0, semantic_similarity * 100)
    
    final_score = 0.4 * hard_match_score + 0.6 * semantic_score
    
    return {
        "score": final_score, "verdict": get_verdict(final_score),
        "hard_match": hard_match_score, "semantic_match": semantic_score,
        "matching_skills": sorted(list(matching_skills)),
        "missing_skills": sorted(list(missing_skills)),
    }

def generate_feedback(results):
    score, missing, matched = results['score'], results['missing_skills'], results['matching_skills']
    feedback = []
    if matched: feedback.append(f"**Great work showcasing skills in:** `{', '.join(matched[:3])}`.")
    if missing:
        feedback.append("\n**Opportunities to strengthen your profile:**")
        suggestions = {
            'tableau': "- For **tableau**: Create a public dashboard project and link it in your resume.",
            'power bi': "- For **power bi**: Create a public dashboard project and link it in your resume.",
            'python': "- For **python**: Ensure your GitHub projects using Python are prominently featured.",
            'sql': "- For **sql**: Include a project where you wrote queries to extract or manipulate data and quantify the outcome."
        }
        for skill in missing[:3]:
            feedback.append(suggestions.get(skill, f"- For **{skill}**: Look for an online course or small project to build this skill."))
    
    feedback.append("\n**Overall Recommendation:**")
    if score < 40: feedback.append("Significant alignment gap. Focus on acquiring foundational skills.")
    elif score < 60: feedback.append("Solid foundation! Add one or two strong projects in missing skill areas.")
    elif score < 90: feedback.append("Strong candidate! Quantify project descriptions to enhance your resume.")
    else: feedback.append("Excellent fit! Prepare for the interview by thinking of specific examples that demonstrate your top skills.")
    return feedback


if 'results' not in st.session_state:
    st.session_state.results = []

st.markdown("<div style='text-align: center;'><h1>Resume Relevance Score Predictor</h1><p>This tool leverages AI to analyze and score a resume's relevance against a job description. It provides a detailed breakdown including keyword matches, contextual similarity, and actionable feedback.</p></div>", unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns(2)
with col1:
    st.header("Step 1: Job Description")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=['pdf', 'docx'], key="jd_uploader")
with col2:
    st.header("Step 2: Student Resume")
    resume_file = st.file_uploader("Upload a single resume (PDF/DOCX)", type=['pdf', 'docx'], key="resume_uploader")

if st.button("Analyze Resume", type="primary", use_container_width=True):
    jd_text, resume_text = extract_text(jd_file), extract_text(resume_file)
    if not jd_text or not resume_text:
        st.warning("Please upload both the Job Description and a resume.")
    else:
        with st.spinner("Analyzing with the new model... This might take a moment!"):
            analysis = analyze_documents(jd_text, resume_text)
            st.session_state.results.append({
                "Resume": resume_file.name, "Relevance Score": f"{analysis['score']:.2f}%",
                "Verdict": analysis['verdict'], "Hard Match": f"{analysis['hard_match']:.2f}%",
                "Semantic Match": f"{analysis['semantic_match']:.2f}%"
            })
        
        st.success("Analysis Complete!")
        st.header("📊 Analysis Results")
        st.subheader(f"Final Verdict: {analysis['verdict']}")
        st.progress(int(analysis['score']), text=f"**Relevance Score: {analysis['score']:.2f}%**")
        st.info(f"**Score Breakdown:** Hard Match: **{analysis['hard_match']:.2f}%** | Semantic Match: **{analysis['semantic_match']:.2f}%**", icon="💡")
        
        st.divider()
        st.subheader("💡 Feedback for Student Improvement")
        for point in generate_feedback(analysis): st.markdown(point)
        
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.subheader("✅ Skills Matched")
            st.write(", ".join(f"`{s}`" for s in analysis['matching_skills']) or "None")
        with res_col2:
            st.subheader("❌ Missing Skills")
            st.write(", ".join(f"`{s}`" for s in analysis['missing_skills']) or "All required skills found!")

st.divider()
st.header("📋 Placement Team Dashboard")
if st.session_state.results:
    st.dataframe(pd.DataFrame(st.session_state.results), use_container_width=True)
    if st.button("Clear Dashboard Results"):
        st.session_state.results = []
        st.rerun()
else:
    st.info("No resumes analyzed yet. The dashboard will populate as you evaluate resumes.")