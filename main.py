import camelot
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy.linalg import norm
import streamlit as st
import pandas as pd


class pdf2text:

    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        temp_table = None
        current_table = None

        texts = []
        tables_df = []

        for table in tables:
            if table.df.index[-1] <= 2:
                temp_table = table.df
            else:
                if temp_table is not None:
                    current_table = pd.concat([temp_table, table.df], ignore_index=True)
                    temp_table = None
                else:
                    current_table = table.df

                tables_df.append(current_table)
                texts.append(current_table.to_string().replace("\\n", ""))

        return texts, tables_df

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        vector = self.model.encode(text)
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = util.pytorch_cos_sim(vector_from_table, vector_from_keyword)
        return cosine

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    table_text, table_df = pdf_parser(pdf_file)
    table_vector = text2vector_parser(table_text)
    keyword_vector = text2vector_parser(keyword)

    cos_sim = []
    for i in range(len(table_vector)):
        cos_sim.append(cosine_sim_parser(table_vector[i], keyword_vector))

    table = table_df[np.argmax(cos_sim)]

    return table

if __name__ == "__main__":
    st.title('BDS HW3a')

    st.write('PDF 1 : 監督式學習 vs. 非監督式學習 vs. 強化學習')
    st.write('PDF 2 : 植物細胞 vs. 動物細胞')

    st.subheader('Choose PDF file')
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])

    st.subheader("Input the keyword")
    keyword = st.text_input("keyword")

    if st.button("Search"):
        table = main(keyword, pdf_file)
        st.write("Result")
        st.write(table)