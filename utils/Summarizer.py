import os
import fitz  # PyMuPDF
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

# --- PDF EXTRACTION AND CLEANING (No changes here) ---

def extract_text_from_pdf(pdf_path):
    """Opens and extracts all text content from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text("text")
        doc.close()
        return full_text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None

def clean_and_split_sentences(sentences):
    """Takes a list of sentences and performs final cleaning and splitting."""
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
    """Cleans raw text, splits it into sentences, and filters out noise."""
    text = " ".join(text.split())
    doc = nlp_model(text)
    initial_sentences = []
    for sent in doc.sents:
        sentence_text = sent.text.strip()
        if len(sentence_text) > 25 and not sentence_text.isupper():
            initial_sentences.append(sentence_text)
    polished_sentences = clean_and_split_sentences(initial_sentences)
    return polished_sentences

def find_top_sentences_by_meaning(sentences, queries, model, top_k=1):
    """
    Finds the most relevant sentences for each query using semantic similarity.
    Defaults to finding the single best sentence (top_k=1).
    """
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

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Loading models...")
    nlp = spacy.load("en_core_web_sm")
    search_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Models loaded.")

    # --- EXPANDED SEARCH QUERIES FOR ALL CATEGORIES ---
    search_queries = {
        "Risks": "This sentence describes potential physical harm, dangers, hazards, or death.",
        "Obligations": "This sentence describes a specific action I must take or a rule I must follow, like wearing equipment or obeying instructions.",
        "Liability": "This sentence describes who is or is not responsible or liable for accidents, and mentions waiving the right to sue.",
        "Duration": "This sentence describes the length, time, or duration of the event or agreement.",
        "Consent": "This sentence describes the participant's confirmation, signature, or explicit agreement to the terms."
    }
    
    data_folder = "data"
    
    for filename in os.listdir(data_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_folder, filename)
            raw_text = extract_text_from_pdf(pdf_path)
            
            if raw_text and len(raw_text) > 100:
                all_sentences = preprocess_text(raw_text, nlp)
                
                query_list = list(search_queries.values())
                # We will now find the single best sentence for each category
                summary_points = find_top_sentences_by_meaning(all_sentences, query_list, search_model, top_k=1)

                # --- NEW STRUCTURED OUTPUT FORMAT ---
                print("\n" + "="*60)
                print(f"📄 Adventure Consent Form Summary: {filename.upper()}")
                print("="*60)

                for i, category_name in enumerate(search_queries.keys()):
                    query_text = query_list[i]
                    top_sent = summary_points.get(query_text)
                    
                    # Print in the desired "Category: [Sentence]" format
                    if top_sent:
                        print(f"- {category_name}: {top_sent[0]}")
                    else:
                        print(f"- {category_name}: Not explicitly found.")
                
                print("="*60)