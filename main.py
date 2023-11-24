import camelot
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from typing import List
from glob import glob


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        self.get_titles(pdf_file)

        tables_info = []
        for table, title in zip(tables, self.titles):
            tables_info.append(
                {
                    "title": title,
                    "table_df": table.df,
                    "table_string": table.df.to_string().replace("\\n", ""),
                }
            )

        return tables_info

    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", "").strip().split(" "))

        self.titles = []
        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    title = word + " " + text[text.index(word) + 1]
                    title = title.replace("ai_tables_", "")
                    self.titles.append(title)


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword).item()


def search_best_table(keyword, pdf_files: List[str]) -> pd.DataFrame:
    # parse pdf tables to string
    pdf_parser = pdf2text()
    tables = []
    for pdf_file_path in pdf_files:
        tables.extend(pdf_parser(pdf_file_path))

    # convert string to vector
    txt2vec = text2vector()
    vectors = []
    for table in tables:
        vectors.append(txt2vec(table["table_string"]))

    # calculate cosine similarity
    cos_sim = cosine_sim()
    cosine_scores = []
    for vector in vectors:
        cosine_scores.append(cos_sim(vector, txt2vec(keyword)))

    # return table with highest similarity
    return tables[np.argmax(cosine_scores)]


def main():
    st.set_page_config(page_title="PDF Table Search Engine", layout="wide")

    st.title("Document Intelligence - PDF Table Search Engine")

    pdf_info = {
        "🤖 1.pdf": "監督式學習、非監督式學習、強化學習相關的資料表格",
        "🔬 2.pdf": "動植物細胞特點、多細胞生物和單細胞生物相關的資料表格",
    }
    for pdf, intro in pdf_info.items():
        st.write(f"{pdf}: {intro}")

    pdf_files = [f for f in glob("docs/*.pdf")]
    selected_pdf = st.multiselect("Select PDF files", pdf_files, default=pdf_files)

    keyword = st.text_input("Enter a keyword")

    if selected_pdf and keyword:
        with st.spinner("🔎 Searching.."):
            table = search_best_table(keyword, selected_pdf)
        st.header("Search Result")
        st.subheader(table["title"])
        st.dataframe(table["table_df"])


if __name__ == "__main__":
    main()
