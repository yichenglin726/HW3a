import camelot
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

from numpy.linalg import norm
import streamlit as st


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        # print(len(tables))
        texts = []
        table_df = []
        for table in tables:
            table_df.append(table.df)
            text = table.df.to_string()
            text = text.replace("\\n", "")
            texts.append(text)
        return texts, table_df


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
        max_sim = 0
        index = 0
        for i in range(len(vector_from_table)):
            dot = np.dot(vector_from_table[i], vector_from_keyword)
            cos_sim = dot / (norm(vector_from_table[i]) * norm(vector_from_keyword))
            print("table ", i, "similarity: ", cos_sim)
            
            if cos_sim > max_sim:
                max_sim = cos_sim
                index = i
        return index


def main(search_query, pdf_file):

    pdf2text_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    table_text, table_df = pdf2text_parser(pdf_file)

    text_vector = text2vector_parser(table_text)
    search_query = text2vector_parser(search_query)

    similar_table_index = cosine_sim_parser(text_vector, search_query)
    table = table_df[similar_table_index]

    return table


if __name__ == "__main__":
    st.title("PDF Table Search")
    # get all pdf in file
    PDF_file = [f for f in os.listdir('docs') if f.endswith('.pdf')]

    # each pdf information        
    st.subheader("PDF 檔案資訊")
    PDF_info = []
    PDF_info.append("監督式學習、非監督式學習、強化學習相關表格")
    PDF_info.append("動植細胞相關表格")

    for i in range(len(PDF_info)):
        st.write(f"PDF {i+1}: {PDF_info[i]}")
    
    # st.subheader("請選擇 PDF 檔案")
    chosed_pdf = st.selectbox("請選擇 PDF 檔案", PDF_file)
    chosed_pdf_path = os.path.join('docs', chosed_pdf)

    search_query = st.text_input("請輸入關鍵字")
    if st.button("搜尋"):
        with st.spinner("搜尋中..."):
            table = main(search_query, chosed_pdf_path)
            st.subheader("搜尋結果")
            st.dataframe(table)
