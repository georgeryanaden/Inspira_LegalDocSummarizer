import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize
import nltk
import os
from datetime import datetime
import spacy
from fpdf import FPDF

nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")

# ========== Helper Functions ==========

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text += page.get_text("text") + "\n"
    return text

def chunk_text(text, max_chunk_tokens=450):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= max_chunk_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_entities(text):
    doc = nlp(text)
    entities = {"PARTIES": set(), "DATES": set(), "MONEY": set()}
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG"]:
            entities["PARTIES"].add(ent.text)
        elif ent.label_ == "DATE":
            entities["DATES"].add(ent.text)
        elif ent.label_ == "MONEY":
            entities["MONEY"].add(ent.text)
    return entities

def summarize_chunks(chunks, model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    summaries = []
    for i, chunk in enumerate(chunks):
        inputs = "summarize: " + chunk.strip()
        tokens = tokenizer(inputs, return_tensors="pt", truncation=True, max_length=512)

        try:
            summary_ids = model.generate(tokens['input_ids'],
                                         max_length=200,
                                         min_length=60,
                                         length_penalty=2.0,
                                         repetition_penalty=1.5,
                                         num_beams=4,
                                         early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            entities = extract_entities(chunk)
            summaries.append((f"Chunk {i+1}", summary, entities))
        except Exception as e:
            summaries.append((f"Chunk {i+1}", f"Error summarizing: {e}", {}))
    return summaries

def generate_pdf(summary_data):
    pdf = FPDF()
    pdf.add_page()

    # Use DejaVuSans for Unicode support
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", "B", 16)

    # Add local logo
    try:
        pdf.image("logo_inspira.jpeg", x=10, y=8, w=50)
    except Exception as e:
        print("⚠️ Could not load local logo:", e)

    pdf.cell(200, 40, "Inspira_LegalDocSummarizer Report", ln=True, align="C")
    pdf.set_font("DejaVu", size=12)

    for title, summary, entities in summary_data:
        pdf.set_font("DejaVu", "B", 12)
        pdf.cell(200, 10, title, ln=True)
        pdf.set_font("DejaVu", size=11)
        pdf.multi_cell(0, 8, summary)

        if entities:
            pdf.set_font("DejaVu", "I", 11)
            pdf.cell(200, 8, "Extracted Entities:", ln=True)
            for key, values in entities.items():
                if values:
                    line = f"  {key}: {', '.join(values)}"
                    pdf.multi_cell(0, 6, line)

        pdf.ln(4)

    filename = f"Inspira_LegalDocSummary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# ========== Streamlit App ==========

st.set_page_config(page_title="Inspira_LegalDocSummarizer", layout="wide")
st.title("📄 Inspira_LegalDocSummarizer")
st.image(r"C:\Users\ASUS\Desktop\Inspira_LegalDoc\logo_inspira.jpeg", width=150)
st.markdown("Upload a legal PDF and get section-wise summaries with named entity highlights.")

uploaded_file = st.file_uploader("Upload your legal PDF file", type=["pdf"])

if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    st.success("Text extracted!")
    st.write(f"Extracted text length: {len(raw_text)} characters")

    chunks = chunk_text(raw_text)
    st.write(f"🧩 Split into {len(chunks)} chunks for summarization")

    if st.button("🔍 Generate Summary"):
        with st.spinner("Summarizing... this may take a moment..."):
            summaries = summarize_chunks(chunks)

        st.success("✅ Summary complete!")
        for title, summary, entities in summaries:
            st.subheader(title)
            st.write(summary)
            if any(entities.values()):
                st.markdown("**Extracted Entities:**")
                for k, v in entities.items():
                    if v:
                        st.markdown(f"- **{k}**: {', '.join(v)}")

        filename = generate_pdf(summaries)
        with open(filename, "rb") as f:
            st.download_button("📥 Download Summary as PDF", f, file_name=filename)
else:
    st.info("Please upload a PDF to begin.")