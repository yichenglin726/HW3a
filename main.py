import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import streamlit as st
import pandas as pd

class pdf2text:
    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df)
        return texts
class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('bert-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)

class cosine_sim:
    def __call__(self, vector_from_table, vector_from_keyword):
        dot_product = np.dot(vector_from_table, vector_from_keyword)
        norm_product = np.linalg.norm(vector_from_table) * np.linalg.norm(vector_from_keyword)
        result = dot_product / norm_product
        return result

def main():
    st.title("PDF 表格搜索引擎")

    pdf_files = [f for f in os.listdir('docs') if f.endswith('.pdf')]
    selected_pdf = st.selectbox("選擇 PDF 文件", pdf_files)

    keyword = st.text_input("輸入關鍵詞")

    if selected_pdf and keyword:
        pdf_file_path = os.path.join('docs', selected_pdf)
        pdf_parser = pdf2text()
        tables = pdf_parser(pdf_file_path)
        txt2vec = text2vector()
        keyword_vector = txt2vec(keyword)

        cos_similarity = cosine_sim()
        max_similarity = -1
        best_table = None
        for i, table in enumerate(tables):
            vector_table = txt2vec(table.to_string())
            similarity_score = cos_similarity(vector_table, keyword_vector)

            st.write(f"Table {i + 1} - Cosine Similarity Score: {similarity_score}")

            if similarity_score > max_similarity:
                max_similarity = similarity_score
                best_table = table

        st.write("Best Matching Table:")
        st.dataframe(best_table)


if __name__ == "__main__":
    main()

