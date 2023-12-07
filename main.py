import camelot
from sentence_transformers import SentenceTransformer
import os
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        return tables

class Text2Vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, table_list, keyword):
        table_text_list = [table.df.to_string(index=False).replace("\\n", "") for table in table_list]
        table_vector = self.model.encode(table_text_list)
        keyword_vector = self.model.encode([keyword])
        return table_vector, keyword_vector

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine_similarity_matrix = cosine_similarity(vector_from_table.reshape(1, -1), vector_from_keyword.reshape(1, -1))
        return cosine_similarity_matrix[0][0]

class GetTable:
    def __init__(self, cosine_sim_calculator, threshold=0.5):
        self.cosine_sim_calculator = cosine_sim_calculator
        self.threshold = threshold

    def __call__(self, tables, keyword):
        text2vec = Text2Vector()
        keyword = keyword.lower()
        table_vector, keyword_vector = text2vec(tables, keyword)
        selected_tables = []
        
        for i, table in enumerate(tables):
            cosine_similarity_score = self.cosine_sim_calculator(table_vector[i], keyword_vector)

            selected_tables.append((cosine_similarity_score, table.df))

        if not selected_tables:
            return None

        selected_tables.sort(key=lambda x: x[0], reverse=True)
        return selected_tables

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)
    
    cosine_sim_calculator = cosine_sim()
    get_table_instance = GetTable(cosine_sim_calculator)
    selected_tables = get_table_instance(tables, keyword)

    return selected_tables

if __name__ == "__main__":
    st.title('BDS hw3a')
    st.subheader("r12922157_林奕丞")
    upload_file = st.file_uploader("Upload your PDF file here", type=['pdf'])
    
    if upload_file is not None:
        pdf_file = os.path.join("docs", upload_file.name)
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())
    
    keyword = st.text_input("Enter your keywords here")
    
    if st.button("Search"):
        tables = main(keyword, pdf_file)

        if tables is not None:
            similarity_score, table = tables[0]
            st.write(table)
