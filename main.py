import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        temp_table = None
        current_table = None

        texts = []
        df_tables = []

        for table in tables:
            if table.df.index[-1] <= 2: # adjacent
                temp_table = table.df
            else:
                if temp_table is not None:
                    current_table = pd.concat([temp_table, table.df], ignore_index=True)
                    temp_table = None
                else:
                    current_table = table.df

                df_tables.append(current_table)
                texts.append(current_table.to_string().replace("\\n", ""))

        titles = self.get_titles(pdf_file)
        return texts, df_tables, titles
    
    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor='stream', pages="all")

        titles = []
        for table in tables:
            text = table.df.to_string().replace("\\n", "").strip().split(' ')
            for word in text:
                if "ai_tables" in word:
                    title = word + " " + text[text.index(word) + 1]
                    titles.append(title)
        return np.sort(list(set(titles)))


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
        np_dot = np.dot(vector_from_table, vector_from_keyword)
        cosine = np_dot / (np.linalg.norm(vector_from_table) * np.linalg.norm(vector_from_keyword))
        return cosine


def main(keyword, pdf_file):
    pdf2text_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    # pdf -> text
    texts, df_tables, titles = pdf2text_parser(pdf_file)

    # text, keyword -> vector
    text_vector = text2vector_parser(texts)
    keyword_vector = text2vector_parser(keyword)

    # cosine similarity
    cosine_similarity = []
    for i in range(len(text_vector)):
        cosine_similarity.append(cosine_sim_parser(text_vector[i], keyword_vector))

    # find the max cosine similarity
    max_index = cosine_similarity.index(max(cosine_similarity))
    table = df_tables[max_index]
    title = titles[max_index]

    return table, title


if __name__ == "__main__":
    st.title("AI - Table Searching With Keyword")
    st.write("AI Table PDF 1: What's AI and Comparing different training 監督式學習 vs. 非監督式學習 vs. 強化學習")
    st.write("AI Table PDF 2: 植物細胞 vs. 動物細胞")

    st.subheader("請選擇一個 PDF 檔案")
    pdf_file = st.selectbox("PDF file", ["docs/1.pdf", "docs/2.pdf"])
    st.subheader("請輸入要檢索的關鍵字：")

    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if submit_button:
        with st.spinner('結果產出中....'):
            table, title = main(keyword, pdf_file)
            st.subheader("Output")
            st.write(title)
            st.dataframe(table)