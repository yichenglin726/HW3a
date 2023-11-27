import camelot
import numpy as np
import pandas as pd
import os
import streamlit as st
import random
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class pdf2text:
    def __init__(self):
        self.tables = []

    def __call__(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        self.tables = [{"df": table.df, "string": table.df.to_string().replace("\\n", "")} for table in tables]
        return self.tables


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)

class cosine_sim:
    def __init__(self):
        pass
    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword) / (np.linalg.norm(vector_from_table)*np.linalg.norm(vector_from_keyword))

def main():
    st.title("PDF Table Search Engine")

    st.write("PDF 1: 監督式學習、非監督式學習、強化學習")
    st.write("PDF 2: 動物細胞和植物細胞、多細胞生物和單細胞生物、多細胞生物細胞膜和植物細胞膜")

    pdf_files = [f for f in os.listdir('docs') if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Select a PDF file", pdf_files)

    keyword = st.text_input("Enter a keyword")

    if selected_pdf and keyword:
        pdf_file_path = os.path.join('docs', selected_pdf)
        with st.spinner('Finding the perfect match for you...'):
            pdf_parser = pdf2text()
            tables = pdf_parser(pdf_file_path)

            txt2vec = text2vector()
            keyword_vector = txt2vec(keyword)
            cosine_scores = [cosine_sim()(txt2vec(table["string"]), keyword_vector) for table in tables]

            best_table = tables[np.argmax(cosine_scores)]["df"]
            st.dataframe(best_table)

if __name__ == "__main__":
    main()
