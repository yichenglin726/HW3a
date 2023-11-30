import camelot
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tmp_path = "temp.pdf"
        with open(tmp_path, "wb") as f:
            f.write(pdf_file.getvalue())

        tables = camelot.read_pdf(tmp_path, pages="all")
        dataframes = []
        for table in tables:
            dataframes.append(table.df)

        Path.unlink(tmp_path)  # remove the tmp file

        return dataframes


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table_embedding, keyword_embedding):
        return cos_sim(table_embedding, keyword_embedding)

def find_table(keyword, pdf_files):
    pdf_parser = pdf2text()
    encoder = text2vector()
    compute_sim = cosine_sim()

    keyword_embed = encoder(keyword)

    best_sim = 0
    best_table_df = None

    for pdf_file in pdf_files:
        table_dfs = pdf_parser(pdf_file)

        for table_df in table_dfs:
            table_embed = encoder(table_df.to_string())
            sim = compute_sim(table_embed, keyword_embed)
            if sim > best_sim:
                best_sim = sim
                best_table_df = table_df

    return best_table_df

class UI:
    def show(self):
        st.title("BDS HW3a: Finding key words in pdf file")
        st.write("B09902111 李哲言")

        pdfs = st.file_uploader("Provide PDF files", accept_multiple_files=True)

        keyword = st.text_input("keyword you want to search")

        st.write("Submit")
        if "alert" in st.session_state and st.session_state.alert != None:
            st.warning(st.session_state.alert)
        st.button("submit", type="primary", on_click=self.on_submit, args=(pdfs, keyword))

        st.divider()

        if "table" in st.session_state:
            st.table(st.session_state.table)

    def on_submit(self, pdfs, keyword):
        st.session_state["alert"] = None
        if pdfs == []:
            st.session_state.alert = "You need to provide at least on PDF files"
            return
        if keyword == "":
            st.session_state.alert = "You need to provide the search key word"
            return

        table_df = find_table(keyword, pdfs)
        st.session_state.table = table_df

if __name__ == "__main__":
    ui = UI()
    ui.show()