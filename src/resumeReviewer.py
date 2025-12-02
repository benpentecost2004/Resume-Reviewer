# pip install streamlit pdfminer.six python-docx spacy
# python -m spacy download en_core_web_sm

import streamlit as st
import re, json, docx, pickle
from pdfminer.high_level import extract_text
import pandas as pd

st.set_page_config(page_title="Resume Reviewer", page_icon="", layout="centered")
st.title("Resume Reviewer")

def extract_text_from_pdf(file):
    return extract_text(file)

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_sections(text):
    text = text.replace('\r', '').replace('\n', '\n')

    sections = {
        'Summary': r'(summary|objective|about me|career focus|career goals)',
        'Experience': r'(experience|employment history|work history)',
        'Skills': r'(skills|technical skills|core competencies)',
        'Education': r'(education|academic background)',
        'Certifications': r'(certifications|licenses)',
        'Projects': r'(projects|personal projects|portfolio)'
    }

    found_sections = {}
    last_pos = None
    last_key = None
    lines = text.splitlines()

    for i, line in enumerate(lines):
        for key, pattern in sections.items():
            if re.search(rf'^\s*{pattern}\b', line.lower()):
                if last_key:
                    found_sections[last_key] = "\n".join(lines[last_pos:i]).strip()
                last_pos = i + 1
                last_key = key

    if last_key:
        found_sections[last_key] = "\n".join(lines[last_pos:]).strip()

    for section in sections.keys():
        found_sections.setdefault(section, "")

    return found_sections

def save_to_json(data, filename_prefix="resume_data"):

    ordered_data = {
        "Summary": data.get("Summary", ""),
        "Experience": data.get("Experience", ""),
        "Skills": data.get("Skills", ""),
        "Education": data.get("Education", ""),
        "Certifications": data.get("Certifications", ""),
        "Projects": data.get("Projects", "")
    }

    filename = f"{filename_prefix}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(ordered_data, f, indent=4, ensure_ascii=False)

    return filename

def load_model():
    with open("resume_data.json", "r") as f:
        data = json.load(f)

    df = pd.DataFrame([data])

    text_cols = ['Summary', 'Experience', 'Skills', 'Education', 'Certifications']
    df[text_cols] = df[text_cols].fillna("")
    df["combined_text"] = df[text_cols].agg(" ".join, axis=1)

    with open('lr_model.pkl', 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(df["combined_text"])

    st.markdown("### Predicted Decision")
    st.write(y_pred[0])

uploaded = st.file_uploader("Upload a resume (PDF or DOCX)", type=["pdf", "docx"])

if uploaded:
    if uploaded.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded)
    elif uploaded.name.endswith(".docx"):
        text = extract_text_from_docx(uploaded)
    else:
        st.error("Please upload a valid PDF or DOCX file.")
        st.stop()

    st.success("Resume uploaded and processed!")
    sections = extract_sections(text)

    filename = save_to_json(sections)

    st.markdown("### Resume Information")
    for section, content in sections.items():
        st.subheader(section)
        st.write(content if content else "— Not found —")
    
    if st.button("Predict"):
        load_model()
        st.success ("Resume has been processed!")  