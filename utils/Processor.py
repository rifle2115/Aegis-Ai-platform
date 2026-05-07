import fitz  # PyMuPDF
import spacy
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline # <-- Re-introducing the transformers library
import streamlit as st

# Cached function to load all models, now including the summarizer
@st.cache_resource
def load_models():
    """Loads all the necessary models and caches them."""
    print("Loading all models... (This may take a moment on first run)")
    nlp = spacy.load("en_core_web_sm")
    search_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Load the summarization model
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    print("All models loaded and cached.")
    return nlp, search_model, summarizer

# Load the models by calling the cached function
nlp, search_model, summarizer = load_models()

# --- Helper functions (extract, clean, preprocess) remain the same ---

def extract_text_from_pdf(pdf_stream):
    try:
        doc = fitz.open(stream=pdf_stream, filetype="pdf")
        full_text = "".join(page.get_text("text") for page in doc)
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading PDF stream: {e}")
        return None

def clean_and_split_sentences(sentences):
    final_sentences = []
    for sentence in sentences:
        sentence = sentence.replace('', '-').replace('•', '-')
        if '-' in sentence:
            split_parts = sentence.split('-')
            for part in split_parts:
                cleaned_part = part.strip()
                if len(cleaned_part) > 20:
                    final_sentences.append(cleaned_part)
        else:
            final_sentences.append(sentence)
    return final_sentences

def preprocess_text(text, nlp_model):
    text = " ".join(text.split())
    doc = nlp_model(text)
    initial_sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 25 and not sent.text.strip().isupper()]
    polished_sentences = clean_and_split_sentences(initial_sentences)
    return polished_sentences

def find_top_sentences_by_meaning(sentences, queries, model, top_k=2):
    if not sentences:
        return {}
    doc_embeddings = model.encode(sentences, convert_to_tensor=True)
    query_embeddings = model.encode(queries, convert_to_tensor=True)
    cosine_scores = util.cos_sim(query_embeddings, doc_embeddings)
    results = {}
    for i, query in enumerate(queries):
        top_results = torch.topk(cosine_scores[i], k=min(top_k, len(sentences)))
        top_indices = top_results.indices.tolist()
        top_sentences = [sentences[idx] for idx in top_indices]
        results[query] = top_sentences
    return results

# --- Main function updated to generate an abstractive summary ---

def generate_summary(pdf_stream):
    """Main function to generate a single, abstractive summary paragraph."""
    search_queries = {
        "Risks": "This sentence describes potential physical harm, dangers, hazards, or death.",
        "Obligations": "This sentence describes a specific action I must take or a rule I must follow, like wearing equipment or obeying instructions."
    }

    raw_text = extract_text_from_pdf(pdf_stream)
    if not raw_text or len(raw_text) < 100:
        return "The PDF could not be read or contains too little text to summarize."
    
    all_sentences = preprocess_text(raw_text, nlp)
    
    # Step 1: Extract MORE important sentences for better context
    query_list = list(search_queries.values())
    # --- THIS IS THE MODIFIED LINE ---
    summary_points = find_top_sentences_by_meaning(all_sentences, query_list, search_model, top_k=5)

    # Step 2: Combine the extracted points into a single text block
    text_to_summarize = ""
    for query, sentences in summary_points.items():
        text_to_summarize += " ".join(sentences) + " "
    
    if len(text_to_summarize.strip()) > 100:
        # Step 3: Generate the final summary (with a slightly longer max_length)
        final_summary = summarizer(text_to_summarize, max_length=200, min_length=70, do_sample=False)
        return final_summary[0]['summary_text']
    else:
        return "Not enough specific content found to generate a final summary."