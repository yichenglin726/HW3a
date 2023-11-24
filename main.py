import camelot
import numpy as np
import pandas as pd
import os
from numpy.linalg import norm
import streamlit as st
import random
from utils import search_messages
from sentence_transformers import SentenceTransformer

class pdf2text:
    def __init__(self):
        """
        self.tables: list of parsed table string.
        It is a list of dict, each dict contains two keys:
            "df": pandas dataframe
            "string": string of the table
        """
        self.tables = []
    def __call__(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        print("Total tables extracted:", tables.n)
        for table in tables:
            self.tables.append({
                "df": table.df,
                "string": table.df.to_string().replace("\\n", "")
            })
        return self.tables
    def print_every_table(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n\n".join(texts)
        text = text.replace("\\n", "")
        print(text)


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword) / (norm(vector_from_table)*norm(vector_from_keyword))


def search_best_table(keyword, pdf_file_path) -> pd.DataFrame:    
    # parse pdf tables to string
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file_path)
    # convert string to vector
    txt2vec = text2vector()
    vectors = []
    for table in tables:
        vectors.append(txt2vec(table["string"]))
    # calculate cosine similarity
    cos_sim = cosine_sim()
    cosine_scores = []
    for vector in vectors:
        cosine_scores.append(cos_sim(vector, txt2vec(keyword)))
    # return table with highest similarity
    return tables[np.argmax(cosine_scores)]["df"]

def main():
    st.title("PDF Table Search Engine")

    pdf_info = {
        '1.pdf': '監督式學習、非監督式學習、強化學習相關的資料表格',
        '2.pdf': '動植物細胞特點、多細胞生物和單細胞生物相關的資料表格',
        # Add more files as needed
    }
    for pdf, intro in pdf_info.items():
        st.write(f"{pdf}: {intro}")


    pdf_files = [f for f in os.listdir('docs') if f.endswith('.pdf')]
    selected_pdf = st.selectbox("Select a PDF file", pdf_files)


    keyword = st.text_input("Enter a keyword")


    if selected_pdf and keyword:
        pdf_file_path = os.path.join('docs', selected_pdf)
        with st.spinner(random.choice(search_messages)):
            df = search_best_table(keyword, pdf_file_path)
        st.dataframe(df)

main()

