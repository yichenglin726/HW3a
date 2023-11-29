import camelot
import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import os

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        rows = []
        for table in tables:
            rows.append(table.df)
        return rows

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False)

def search_text_in_pdf(text, pdf):
    t2v = text2vector()
    content = pdf2text()(pdf)
    text_vec = t2v(text)

    opt = (0, None)
    for row in content:
        vec = text2vector()(row.to_string())
        sim = cos_sim(vec, text_vec)
        print("sim", sim, "row", row)
        opt = max(opt, (sim, row))
    
    return opt[1]
        
class UI:
    def __init__(self):
        st.title("BDS Hw3a")
        st.write("B09902102 陳冠辰")
        pdfs = st.file_uploader("PDF to search", accept_multiple_files=True)
        text = st.text_input("Keyword to search", "")

        for pdf in pdfs:
            with open("source.pdf", 'wb') as f:
                f.write(pdf.read())
            if st.button("Search"):
                st.write(search_text_in_pdf(text, "source.pdf"))

if __name__ == "__main__":
    ui = UI()
