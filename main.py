import camelot
from numpy import dot, argmax
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
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
            
            words_of_texts = [stream.df.to_string().replace("\\n", "").strip().split(" ") for stream in streams]
            for text in words_of_texts:
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

    def __call__(self, vec_table, vec_keyword):
        return dot(vec_table, vec_keyword) / (norm(vec_table) * norm(vec_keyword))

def search(keywords, pdf_files):
    st.session_state.null = None
    if pdf_files == "" or keywords == "":
        st.session_state.null = "PDF or Keywords are null"
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
    
    idx = argmax(sim_tables)
    return titles[idx], dfs[idx]

if __name__ == "__main__":
    st.title("Hw3a - Document Intelligence")
    st.subheader("R12922201 梁修維")
    st.divider()

    st.subheader("1. Upload PDF file(s):")
    files = st.file_uploader("Upload PDF files.", accept_multiple_files=True)

    st.subheader("2. Input Search Query Keyword(s):")
    query = st.text_input("Search Query")

    if st.button("Search"):
        title, df = search(query, files)
        st.write(title)
        st.table(df)