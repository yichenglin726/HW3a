import camelot
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, files):
        dfs, titles, texts = [], [], []
        for file in files:
            tables = camelot.read_pdf(f"docs/{file.name}", pages="all")
            streams = camelot.read_pdf(f"docs/{file.name}", flavor="stream", pages="all")
            for table in tables:
                dfs.append(table.df)
                texts.append(table.df.to_string().replace("\\n", ""))
            words_string = [stream.df.to_string().replace("\\n", "").strip().split(" ") for stream in streams]
            for text in words_string:
                for word in text:
                    if "ai_tables" in word:
                        title = f"{word} {text[text.index(word) + 1]}".rstrip()
                        titles.append(title)
        return dfs, titles, texts

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
    
    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False)

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_table, vector_keyword):
        return util.pytorch_cos_sim(vector_table, vector_keyword)

def search(keywords, pdf_files):
    st.session_state.null = None
    if pdf_files == "" or keywords == "":
        st.session_state.null = "Error: please input PDF files or type in your search keywords."
        return
    
    pdf_parser = pdf2text()
    txt2vec = text2vector()
    cos_sim = cosine_sim()
    
    dfs, titles, texts = pdf_parser(pdf_files)
    vec_keyword = txt2vec(keywords)
    sim_tables = []
    for table in texts:
        vec_table = txt2vec(table)
        sim_tables.append(cos_sim(vec_table, vec_keyword))
    max_index = np.argmax(sim_tables)
    return titles[max_index], dfs[max_index]

if __name__ == "__main__":
    st.title("BDS HW3a")
    st.subheader("R11941090 王立愷", divider='blue')
    st.subheader("First, choose PDF files from your computer.")
    input_files = st.file_uploader("Upload PDF files:", accept_multiple_files=True)
    st.subheader("Next, input your search keywords.")
    search_query = st.text_input("Search query:")

    if st.button("Search"):
        title, df = search(search_query, input_files)
        st.write(title)
        st.table(df)