import camelot
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        self.get_titles(pdf_file)

        texts = []
        for table, title in zip(tables, self.titles):
            table.df.replace(r'\n', '', regex=True, inplace=True)
            texts.append({
                "title": title,
                "df": table.df,
                "string": table.df.to_string()
            })
        # text = "\n".join(texts)
        # text = text.replace("\\n", "")
        return texts

    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")

        texts = []
        for table in tables:
            texts.append(table.df.to_string().replace(
                "\\n", "").strip().split(' '))
        self.titles = []
        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    title = word + " " + text[text.index(word) + 1]
                    self.titles.append(title)


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer(
            'distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        dot_product = np.dot(vector_from_table, vector_from_keyword)
        product_of_vector_norm = norm(
            vector_from_table) * norm(vector_from_keyword)
        return dot_product / product_of_vector_norm


def main(keyword, pdf_file):
    # Get the texts
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)

    # Text To Vector
    txt2vec = text2vector()
    vectors = []
    for table in table_text:
        vectors.append(txt2vec(table["string"]))

    # Compute cosine similarity
    cosine_similarity = cosine_sim()
    similaritys = []
    for vector_from_table in vectors:
        vector_from_keyword = txt2vec(keyword)
        similarity = cosine_similarity(
            vector_from_table, vector_from_keyword)
        similaritys.append(similarity)
    targetTableTitle = table_text[np.argmax(similaritys)]["title"]
    targetTable = table_text[np.argmax(similaritys)]["df"]
    return targetTableTitle, targetTable


if __name__ == "__main__":
    # UI
    # pdf_path = input("PDF file path:")
    # keyword = input("Searching Keywords:")
    # table = main(keyword, pdf_path)
    st.set_page_config(
        page_title="AI Education Tool",
        page_icon=":robot_face:",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    st.title("AI Education Tool 🤖")

    st.subheader("📄Select a PDF file: ")

    # ADD files if needed
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])

    st.subheader("🔎Input a keyword: ")
    keyword = st.text_input("Input Keywords")
    

    if (pdf_file and keyword):
        with st.spinner('Searching....'):
            title, table_text = main(keyword, pdf_file)
            st.subheader(title)
            st.dataframe(table_text)
