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
        dfs = []
        for table in tables:
            dfs.append(table.df)

        Path.unlink(tmp_path)  # remove the tmp file

        return dfs


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

def find_table(keyword, pdf_files, pbar):
    pdf_parser = pdf2text()
    embedder = text2vector()
    get_sim = cosine_sim()

    keyword_embed = embedder(keyword)

    best_sim = 0
    best_table_df = None

    for i, pdf_file in enumerate(pdf_files):
        table_dfs = pdf_parser(pdf_file)

        curr_progress = (i * 100) // len(pdf_files)
        progress_step = 100 // len(pdf_files) // len(table_dfs)

        msg = "Seaching..." 
        pbar.progress(curr_progress, msg)

        for j, table_df in enumerate(table_dfs):
            table_text_embed = embedder(table_df.to_string())
            sim = get_sim(table_text_embed, keyword_embed)
            if sim > best_sim:
                best_sim = sim
                best_table_df = table_df

            msg = f"{j + 1}/{len(table_dfs)} table in {pdf_file.name}"
            pbar.progress(curr_progress + (j + 1) * progress_step, msg)

    pbar.progress(100)

    return best_table_df

class UI:
    def show(self):
        st.title("R12921036 HW3a")

        pdfs = st.file_uploader("Upload pdf files.", accept_multiple_files=True)

        query = st.text_input("Input keywords.")

        if "alert" in st.session_state and st.session_state.alert != None:
            st.warning(st.session_state.alert)
        st.button("Search", type="primary", on_click=self.on_submit, args=(pdfs, query))
        self.pbar = st.empty()

        st.divider()

        if "table" in st.session_state:
            st.table(st.session_state.table)

    def on_submit(self, pdfs, query):
        st.session_state["alert"] = None
        if pdfs == []:
            st.session_state.alert = "Error: No pdf files"
            return
        if query == "":
            st.session_state.alert = "Error"
            return

        table_df = find_table(query, pdfs, self.pbar)
        st.session_state.table = table_df

if __name__ == "__main__":
    ui = UI()
    ui.show()