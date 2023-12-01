import camelot
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        texts = []
        df = []
        tables = camelot.read_pdf(f'docs/{pdf_file.name}', pages="all")

        for i in tables:
            df.append(i.df)
            texts.append(i.df.to_string().replace("\\n", "").replace(" ", ""))
        
        return df,texts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table, keyword):
        return (np.dot(table, keyword)) / (norm(table) * norm(keyword))

def search_pdf(keyword, pdf_file):
    st.session_state.null = None
    if (pdf_file == "" or keyword == ""):
        st.session_state.null = "PDF or Keyword is null"
        return

    pdf_parser = pdf2text()
    text2vec = text2vector()
    cos_sim = cosine_sim()

    table_df,table_text = pdf_parser(pdf_file)
    keyword_vector = text2vec(keyword)

    relevant_tables = []

    for text in table_text:
        relevant_tables.append(cos_sim(text2vec(text), keyword_vector))

    st.table(table_df[np.argmax(relevant_tables)])

if __name__ == "__main__":
    st.title("Welcome to My Artificial Intelligence Table Searcher ( ˘͈ ᵕ ˘͈♡)")
    st.subheader("🎀by R12922176 康甜甜 碩一🎀")
    st.divider()

    st.subheader("1. Choose your Table File :")
    file = st.file_uploader("Provide PDF files that may contain interested tables.", accept_multiple_files=False)

    st.subheader("2. Input your keyword to search query :")
    text = st.text_input("Please input your keyword")

    st.button("Search", type="primary", on_click=search_pdf(text, file))
    if "null" in st.session_state and st.session_state.null != None:
        st.warning(st.session_state.null)