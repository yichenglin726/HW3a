import camelot
import streamlit as st
import numpy as np
from sentence_transformers import util, SentenceTransformer


class pdf2text:
    def __init__(self):
        self.tables = []


    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        for table in tables:
            table = {
                'df': table.df,
                'text': table.df.to_string().replace("\\n", ""),
            }
            self.tables.append(table)
        return self.tables


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)


def searchTable(keyword, pdf_file):
    # Initialization
    cos_sim = cosine_sim()
    pdf_parser = pdf2text()
    vectorizer = text2vector()

    # Get the tables
    tables = pdf_parser(pdf_file)

    # Get the similarity scores
    k = vectorizer(keyword)
    sim = [cos_sim(vectorizer(t['text']), k) for t in tables ]

    return tables[np.argmax(sim)]


if __name__ == "__main__":

    # Page title
    st.set_page_config(page_title='AI Table')
    st.title(':cherry_blossom: :rainbow[AI Table]')

    # Selection
    st.header('Please select a pdf:')
    default_pdf = ["1.pdf", "2.pdf"]
    option = {
        "label": 'Select a pdf file',
        "options": default_pdf,
    }
    selection = st.selectbox(**option)
    selected_pdf = f"docs/{selection}"

    # Search
    st.header('Input Keyword:')
    keyword = st.text_input("Keyword", "非監督式學習的應用")
    click = st.button("Search")

    # Show the results
    if (click):
        st.header('Output:')
        with st.spinner('In progress......'):
            table = searchTable(keyword, selected_pdf)
        st.write("Table: ")
        st.dataframe(table['df'])