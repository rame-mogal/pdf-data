import os
import json
import re
import fitz  # PyMuPDF
import tempfile
import pandas as pd
import streamlit as st
from PIL import Image
from dotenv import load_dotenv
import openai
from paddleocr import PaddleOCR
import numpy as np
from PyPDF2 import PdfReader
from collections import defaultdict

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Initialize PaddleOCR (Disable angle classifier to avoid error)
ocr_model = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False)

# Streamlit UI
st.set_page_config(page_title="PDF Info Extractor", page_icon="🔍")
st.title("🔍 Boltware PDF Information Extractor")
st.markdown("Upload a PDF. If it's scanned, OCR + GPT-3.5 will extract structured data. If it's text-based, data is extracted directly.")

uploaded_file = st.file_uploader("📄 Upload your PDF", type=["pdf"])
selected_fields = st.multiselect("Select fields to extract:", ["Firm Name", "Address", "Date", "Turnover", "Contact Details"])

# Check if PDF is scanned
def is_scanned_pdf(file_path: str) -> bool:
    try:
        reader = PdfReader(file_path)
        text = reader.pages[0].extract_text()
        return not bool(text and text.strip())
    except Exception as e:
        st.warning(f"PDF check failed: {e}")
        return True  # Assume scanned if error occurs

# Query OpenAI
def query_openai(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"OpenAI API Error: {e}")
        return None

# Prompt for GPT
def build_prompt(text, selected_fields):
    fields_string = "\n- ".join(selected_fields)
    json_fields = ",\n  ".join([f'"{field}": ["..."]' for field in selected_fields])
    return f"""
Extract the following fields from the text. If multiple values exist for a field, return them all as a list:

- {fields_string}

Text:
{text}

Respond ONLY in this JSON format:
{{
  {json_fields}
}}
"""

# OCR text from image
def paddle_ocr_text(image: Image.Image):
    img_array = image.convert("L")
    results = ocr_model.ocr(np.array(img_array), cls=False)
    full_text = ""
    for line in results[0]:
        text = line[1][0]
        full_text += text + "\n"
    return full_text

# Extract text from text-based PDF
def extract_text_from_text_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(
        page.extract_text() or "" for page in reader.pages
    )

# Extract text from scanned PDF via OCR
def extract_text_from_scanned_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    full_text = ""
    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap(dpi=100)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img = img.convert("L")
            page_text = paddle_ocr_text(img)
            full_text += page_text + "\n"
        except Exception as e:
            st.warning(f"OCR failed on page {i + 1}: {e}")
    return full_text

# Run main logic
if uploaded_file and selected_fields:
    with st.spinner("🔍 Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_pdf_path = tmp_file.name

        scanned = is_scanned_pdf(tmp_pdf_path)
        st.info("Detected **scanned PDF**. Using OCR." if scanned else "Detected **text-based PDF**. Extracting text directly.")

        extracted_text = extract_text_from_scanned_pdf(tmp_pdf_path) if scanned else extract_text_from_text_pdf(tmp_pdf_path)

        max_len = 3000
        chunks = [extracted_text[i:i + max_len] for i in range(0, len(extracted_text), max_len)]

        combined_data = defaultdict(set)

        for chunk in chunks:
            prompt = build_prompt(chunk, selected_fields)
            response = query_openai(prompt)
            if response:
                match = re.search(r'\{.*\}', response, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        for key, values in data.items():
                            if isinstance(values, list):
                                combined_data[key].update(values)
                            elif values:
                                combined_data[key].add(values)
                    except json.JSONDecodeError:
                        st.warning("⚠️ Invalid JSON in response.")

        # Format as rows
        max_rows = max((len(v) for v in combined_data.values()), default=0)
        table_data = []
        for i in range(max_rows):
            row = {}
            for key in selected_fields:
                values = list(combined_data.get(key, []))
                row[key] = values[i] if i < len(values) else ""
            table_data.append(row)

        if table_data:
            df = pd.DataFrame(table_data)
            st.subheader("✅ Extracted Information (Multiple Values)")
            st.dataframe(df)
            st.download_button("⬇️ Download as JSON", json.dumps(table_data, indent=2), file_name="extracted_info.json")
            st.download_button("⬇️ Download as CSV", df.to_csv(index=False), file_name="extracted_info.csv")
        else:
            st.error("❌ No valid data extracted.")
