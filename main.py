import camelot
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='lattice', pages="all")
        texts = []
        table_df = []

        for table in tables:
            # each table_df is a datafram of a table
            table_df.append(table.df)
            text = table.df.to_string(index=False)
            text = text.replace("\\n", "")
            texts.append(text)
        
        return texts, table_df


class text2vector:
    def __init__(self):
        # pass
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


    def __call__(self, text):
        # pass
        embeddings = self.model.encode(text)
        return embeddings


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        # pass
        dot_product = np.dot(vector_from_table, vector_from_keyword)
        norm_t = np.linalg.norm(vector_from_table)
        norm_k = np.linalg.norm(vector_from_keyword)
        similarity = dot_product / (norm_t * norm_k)
        return similarity


def main(keyword, pdf_file):
    #print("hello")
    # pdf to text
    pdf_parser = pdf2text()
    table_text, table_dfs = pdf_parser(pdf_file)
    # text to vector
    vector_parser = text2vector()
    keyword_vector = vector_parser(keyword)
    #table_text_vector = vector_parser(table_text)
    

    #cosine similarity
    cos_sim_calculator = cosine_sim()
    similarity_scores = []

    for text in table_text:
        # text = table.to_string(index=False)
        table_vector = vector_parser(text)
        #keyword_vector = vector_parser(keyword)
        similarity = cos_sim_calculator(table_vector, keyword_vector)
        similarity_scores.append(similarity)
        # print("sim:")
        # print(similarity)

    # Find the index of the table with the highest similarity score
    max_sim_index = np.argmax(similarity_scores)
    # print("index:")
    # print(max_sim_index)
    return table_text[max_sim_index], table_dfs[max_sim_index]
    #print(table_text)
    #return table_text, table
    # return table


if __name__ == "__main__":
    #main("keyword", widget.choice)
    #main("keyword", "docs/2.pdf")

    st.title('SEARCHING TABLE by :red[_kEYWORD_] !')
    st.header('SELECT A PDF:')
    selected_pdf = st.selectbox("PDF 1 : What's AI and Comparing different training, PDF 2 : 植物細胞 vs. 動物細胞", ["PDF 1", "PDF 2"])
    if (selected_pdf == "PDF 1") :
        selected_pdf = "docs/1.pdf"
    else :
        selected_pdf = "docs/2.pdf"


    st.header('INPUT A KEYWORD:')
    keyword = st.text_input("KEYWORD", "非監督式學習的應用")

    click = st.button("CHECK")

    if (click):
        st.header('OUTPUT:')
        st.write('Please wait a moment...')
        text, df = main(keyword, selected_pdf)
        st.write("TABLE: ")
        st.dataframe(df)

        
