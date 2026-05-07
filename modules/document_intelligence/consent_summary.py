import streamlit as st
from utils.Processor import generate_summary

# --- App Title and Description ---
st.title("📄 Adventure Form Summarizer")
st.write("Upload a liability waiver or consent form in PDF format, and this app will generate a concise summary paragraph for you.")

# --- PDF Upload ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# --- Processing and Displaying the Summary ---
if uploaded_file is not None:
    with st.spinner('Reading the document and generating summary...'):
        pdf_bytes = uploaded_file.getvalue()
        
        # This function now returns a single string
        final_summary_text = generate_summary(pdf_bytes)

    st.success("Summary Generated!")

    # Display the final summary paragraph
    st.subheader(f"AI-Generated Summary for: {uploaded_file.name}")
    st.write(final_summary_text)
