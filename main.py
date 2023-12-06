import camelot
import streamlit as st
from text2vec import SentenceModel
import numpy as np
import sys


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        self.tables = [{"df": table.df, "string": table.df.to_string().replace("\\n", "")} for table in tables]
        return self.tables


class text2vector:
    def __init__(self):
        self.model = SentenceModel('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table, vector_from_keyword) / (np.linalg.norm(vector_from_table) * np.linalg.norm(vector_from_keyword))


def main():
    #streamlit description
    st.title("Hw3a: Stage-A Document Intelligence")
    
    keyword = st.text_input("Enter the keyword:")

    # result display
    st.write("Search Result")

    #pdf2text
    pdf_file = sys.argv[1]
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)

    #text2vec
    encoder = text2vector()
    vectors = [encoder(table["string"]) for table in tables]

    #calculate max similarity
    cos_sim_fn = cosine_sim()
    keyword = encoder(keyword)
    scores = [cos_sim_fn(vector, keyword) for vector in vectors]
    target = np.argmax(scores)

    df = tables[target]["df"]
    st.dataframe(df)

if __name__ == "__main__":
   main()
