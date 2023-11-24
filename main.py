import camelot
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tmp_path = "tmp.pdf"
        with open(tmp_path, "wb") as f:
            f.write(pdf_file.getvalue())

        tables = camelot.read_pdf(tmp_path, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())

        Path.unlink(tmp_path)  # remove the tmp file

        return texts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cos_sim(vector_from_table, vector_from_keyword)

def find_table(keyword, pdf_files):
    pdf_parser = pdf2text()
    embedder = text2vector()
    get_sim = cosine_sim()

    keyword_embed = embedder(keyword)

    best_sim = 0
    best_table_text = None

    for pdf_file in pdf_files:
        table_texts = pdf_parser(pdf_file)
        for table_text in table_texts:
            table_text_embed = embedder(table_text)
            sim = get_sim(table_text_embed, keyword_embed)
            if sim > best_sim:
                best_sim = sim
                best_table_text = table_text

    return best_table_text

class UI:
    def show(self):
        st.title("BDS HW3a: Document Intelligence")
        st.write("R12922060 官澔恩")

        st.header(":sparkles: Find Relevant Tables Easily in 3 Steps")
        st.subheader("Step 1")
        pdfs = st.file_uploader("Provide PDF files that may contain interested tables.", accept_multiple_files=True)

        st.subheader("Step 2")
        query = st.text_input("Type in descriptions of the table you want to find.")

        st.subheader("Step 3")
        st.write("Hit the button, and there you go!")
        if "alert" in st.session_state and st.session_state.alert != None:
            st.warning(st.session_state.alert)
        st.button("Let's Go!", type="primary", on_click=self.on_submit, args=(pdfs, query))

        st.divider()

        if "table" in st.session_state:
            st.text(st.session_state.table)

    def on_submit(self, pdfs, query):
        st.session_state["alert"] = None
        if pdfs == []:
            st.session_state.alert = "You miss the PDF files!"
            return
        if query == "":
            st.session_state.alert = "Tell me what content you are interested in."
            return

        with st.spinner("test"):
            table_text = find_table(query, pdfs)
        st.session_state.table = table_text

if __name__ == "__main__":
    ui = UI()
    ui.show()
