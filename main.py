import camelot
from sentence_transformers import SentenceTransformer
import os
import numpy as np
from numpy.linalg import norm
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            table_text = table.df.to_string().replace("\\n", "")
            table_text = f"一個表格，表格內容包含：\n{table_text}"
            texts.append(table_text)
        return texts


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def __call__(self, table_list, keyword):
        table_vector = self.model.encode(table_list)
        keyword_vector = self.model.encode(keyword)
        return table_vector, keyword_vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        similar_measure = np.dot(vector_from_table, vector_from_keyword)/(norm(vector_from_table, axis=1)*norm(vector_from_keyword))
        return similar_measure


class get_table:
    def __init__(self):
        pass

    def __call__(self, table_list, keyword):
      table_vector, keyword_vector = text2vector()(table_list, keyword)
      cosine_similarity = cosine_sim()(table_vector, keyword_vector)
      index = np.array(cosine_similarity).argmax()
      return table_list[index].removeprefix("一個表格，表格內容包含：\n")


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_list = pdf_parser(pdf_file)
    table = get_table()(table_list, keyword)

    return table

if __name__ == "__main__":
    st.title('BDS hw 3-a 林緯翔')
    st.write("請上傳pdf檔案")

    upload_file = st.file_uploader("Upload PDF", type=['pdf'])
    
    if upload_file is not None:
        st.write("upload success")
        pdf_file = os.path.join("docs", upload_file.name)
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())
    
    st.write("請輸入關鍵字")
    keyword = st.text_input("Search query", "非監督式學習的應用")
    
    if st.button("Search"):
        table = main(keyword, pdf_file)
        st.write("Output:")
        st.write(table)