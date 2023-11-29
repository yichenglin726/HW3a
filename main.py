import camelot
import numpy as np
import pandas as pd
import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# pdf to text
class pdf2text:
    def __init__(self):
        self.tables = []

    def __call__(self, pdf_file_path: str):
        tables = camelot.read_pdf(pdf_file_path, pages="all")
        self.tables = [{"df": table.df, "string": table.df.to_string().replace("\\n", "")} for table in tables]
        return self.tables

# test to vector
class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)

# cosine similarity
class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity([vector_from_table], [vector_from_keyword])[0][0]

def main():

    # streamlit words
    st.title("PDF Content Search")
    st.markdown("1.pdf: 監督式學習、非監督式學習、強化學習相關的資料表格\n\n2.pdf: 動植物細胞特點、多細胞生物和單細胞生物相關的資料表格")

    # select PDF file
    pdf_files = [f for f in os.listdir('docs') if f.endswith('.pdf')]
    pdf_selected = st.selectbox("Select a PDF file", pdf_files)

    # input prompt bar 
    keyword = st.text_input("Please Enter the keyword")

    # result display
    st.write("Search Result")

    # Content search
    if pdf_selected and keyword:

        pdf_file_path = os.path.join('docs', pdf_selected)

        # parse pdf tables to string
        pdf_parser = pdf2text()
        tables = pdf_parser(pdf_file_path)

        # convert text to vector
        txt2vec = text2vector()
        vectors = [txt2vec(table["string"]) for table in tables]    
        
        # calculate cosine similarity
        cos_sim = cosine_sim()
        cosine_scores = [cos_sim(vector, txt2vec(keyword)) for vector in vectors]   

        df = tables[np.argmax(cosine_scores)]["df"]
        st.dataframe(df)

if __name__ == "__main__":
    main()

