import os
import camelot
from sentence_transformers import SentenceTransformer, util
import streamlit as st
import torch


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text


def pdf_to_tables(pdf_path):
    return [
        table.df.replace(r"\s", "", regex=True)
        for table in camelot.read_pdf(pdf_path, pages="all")
    ]


class text2vector:
    def __init__(self):
        self.__model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    def __call__(self, text):
        return self.__model.encode(text)


@st.cache_resource
def cached_text2vector():
    return text2vector()


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_keyword, vector_from_table)


def main():
    model = cached_text2vector()

    tables = []
    for root, dirs, files in os.walk("./docs"):
        for file in files:
            if file.endswith(".pdf"):
                tables.extend(pdf_to_tables(os.path.join(root, file)))
    table_embeddings = model([table.to_string() for table in tables])

    st.text_input("Query", key="query")
    if not st.session_state.query:
        "The query is empty."
    else:
        query_embedding = model(st.session_state.query)

        cos_scores = cosine_sim()(table_embeddings, query_embedding)[0]

        top_k = st.number_input(
            "Number of results", min_value=1, max_value=len(tables), value=1
        )
        top_results = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_results[0], top_results[1]):
            tables[idx]

        with st.expander("Scores"):
            for idx, cos_score in enumerate(cos_scores):
                cos_score
                tables[idx]
        

    with st.expander("All tables"):
        for table in tables:
            table


if __name__ == "__main__":
    main()
