import camelot
from sentence_transformers import SentenceTransformer
import numpy as np
from numpy.linalg import norm
import pandas as pd
import streamlit as st
import os


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
            
            # table.df.index[-1]: column of table

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
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding
    
class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = np.dot(vector_from_table,vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))
        return cosine
    
def UI():
    st.title('巨量資料系統 HW3a')
    st.write("請上傳你的PDF檔案")
    upload_file = st.file_uploader("Upload YOUR PDF File", type=['pdf'])
    if upload_file is not None:
        st.write("你上傳的檔案為")
        st.write(upload_file.name)
        #pdf_file = os.path.join("docs", upload_file.name)
        pdf_file = upload_file.name
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())

    st.write("請輸入你想要搜尋的關鍵字")
    #keyword = st.text_input("keyword", "非監督式學習的應用")

    keyword = st.text_input("Keyword")
    submit_button = st.button("Search")

    if submit_button:
        table, title = main(keyword, pdf_file)
        st.write("結果")
        st.write(title)
        st.dataframe(table)    
    
    
def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()


    texts, df_tables, titles = pdf_parser(pdf_file)
    #print(texts)
    #print(df_tables)
    #print(titles)

    # text to vector
    table_vector = text2vector_parser(texts)
    keyword_vector = text2vector_parser(keyword)
    #print(table_vector)
    #print(keyword_vector)
    
    # cosine similarity
    cosine_similarity = []
    for i in range(len(table_vector)):
        cosine_similarity.append(cosine_sim_parser(table_vector[i], keyword_vector))
    #print(cos_sim)
    

    # find the max cosine similarity
    max_index = cosine_similarity.index(max(cosine_similarity))
    table = df_tables[max_index]
    title = titles[max_index]
    print(table)
    print(title)

    return table, title 
    
    
if __name__ == "__main__":
    #main("keyword", "1.pdf")
    #main("keyword", "2.pdf")
    UI()
