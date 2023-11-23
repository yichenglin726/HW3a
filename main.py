import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from tabulate import tabulate
import streamlit as st
import os

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")

        tiles = self.extract_table_title(pdf_file)

        display_tables = []
        for table, title in zip(tables, tiles):
            display_table = {"title": title, "table": table.df}
            display_tables.append(display_table)

        return display_tables
    
    def extract_table_title(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")

        texts = []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", "").strip().split(' '))

        ai_table_titles = []

        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    ai_table_titles.append(word+" "+text[text.index(word)+1])

        ai_table_titles = list(set(ai_table_titles))
        ai_table_titles = np.sort(ai_table_titles)

        return ai_table_titles


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
        dot_product = np.dot(vector_from_table, vector_from_keyword)
        norm_table =  np.linalg.norm(vector_from_table, axis=1)
        norm_keyword = np.linalg.norm(vector_from_keyword)
        similarity = dot_product / (norm_table * norm_keyword)
        
        max_similarity = np.max(similarity)
        if max_similarity > 0.9:
            similarity_socre = max_similarity
        else:
            similarity_socre = np.mean(-np.sort(-similarity)[:8])

        return similarity_socre


def main(keyword_=None, pdf_file_=None):
    pdf_parser = pdf2text()
    text2vector_function = text2vector()
    similarity_function = cosine_sim()

    PDF_FOLDER = "docs"

    st.title("檢索 PDF 中的表格")
    uploaded_file = st.file_uploader("請選擇 PDF 文件", type=["pdf"])

    if uploaded_file is not None:
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        keyword = st.text_input("請輸入要檢索的關鍵字：", key=uploaded_file.name)
        if keyword == "":
           pass  
        elif keyword is not None:
            tables = pdf_parser(file_path)
            keyword_vector = text2vector_function(keyword)

            max_score=0
            for table in tables:
                table_title, table_df = table["title"], table["table"]
                table_texts = table_title + "\n" + table_df.to_string().replace("\\n", "")
                words = table_texts.strip().split('\n')
                
                words = [word for line in words for word in line.split()]
                word_vectors = text2vector_function(words)

                score = similarity_function(word_vectors, keyword_vector)
                if score > max_score:
                    max_score = score
                    best_match_table = {"title": table_title, "table": table_df}

            
            best_match_table["table"].columns = best_match_table["table"].loc[0, 0:]
            best_match_table["table"] = best_match_table["table"].drop([0])
            st.write(best_match_table["title"])
            st.markdown(best_match_table["table"].style.hide(axis="index").to_html(), unsafe_allow_html=True)

    # return best_match_table


if __name__ == "__main__":

    # main("監督式學習的優缺點", "docs/1.pdf")
    # main("動物細胞特點", "docs/2.pdf")
    main()
