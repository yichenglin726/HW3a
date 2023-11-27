import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for i, table in enumerate(tables):
            text = table.df.to_string()
            text = text.replace("\\n", "")
            texts.append(text)
        return texts, tables

    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")

        table_titles = []
        for table in tables:
            text = table.df.to_string().replace("\\n", "").strip().split(' ')
            for i, word in enumerate(text):
                if "ai_tables" in word:
                    title = word + " " + text[i + 1]
                    table_titles.append(title)
        return table_titles


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        embeddings = self.model.encode(text)
        return embeddings


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        vec1 = np.array(vector_from_table)
        vec2 = np.array(vector_from_keyword)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



def main(keywords, pdf_file):
    pdf_parser = pdf2text()
    cosine = cosine_sim()
    encoder = text2vector()

    table_text, tables = pdf_parser(pdf_file)
    titles = pdf_parser.get_titles(pdf_file)

    table_vec = encoder(table_text)
    title_vec = encoder(titles)

    keywords_vec = encoder(keywords)


    table_sim = []
    title_sim = []
    for vec in table_vec:
        table_sim.append(cosine(vec, keywords_vec))
    
    for vec in title_vec:
        title_sim.append(cosine(vec, keywords_vec))

    table_idx = np.argmax(table_sim)
    title_idx = np.argmax(title_sim)
    #table = tables[idx]
    #print(titles[idx])
    return tables[table_idx].df, titles[title_idx]


if __name__ == "__main__":
    st.title("BDS HW3a")
    st.subheader("Select PDF to search")
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])
    st.subheader("Input keywords")
    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if (submit_button and pdf_file and keyword):
        with st.spinner('Searching'):
            table, title = main(keyword, pdf_file)
            #st.subheader(title)
            st.write(table)
