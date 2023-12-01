import streamlit 
import camelot
import os
from sentence_transformers import util, SentenceTransformer
import tempfile

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):

        tables = camelot.read_pdf(pdf_file, pages="all")
      
        texts_return = [] 
        tables_return = []

        for table in tables:
            tables_return.append(table.df)
            text = table.df.to_string()
            texts_return.append(text.replace("\\n", ""))
           
        return texts_return, tables_return

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False) 

class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword)


def main(keyword, pdf_file):
    pdf_parser = pdf2text() 
    texts, tables = pdf_parser(pdf_file)

    vectorizer = text2vector() 
    keyword_vector = vectorizer(keyword)

    simularity_counter = cosine_sim()

    max_simularity = 0
    ret_table = None
    
    for table, text in zip(tables, texts):
        keyword_vector, text_vector = vectorizer([keyword, text])
        
        simularity = simularity_counter(text_vector, keyword_vector)
        
        if simularity > max_simularity:
            ret_table = table
            max_simularity = simularity
    
    return ret_table


if __name__ == "__main__":

    streamlit.title('HW3.a PDF Search')

    uploaded_file = streamlit.file_uploader("上傳PDF檔案", type=["pdf"])
    
    # 輸入文字
    user_input = streamlit.text_input('輸入關鍵字') 

    # 顯示表單
    if streamlit.button('生成表單'):
        if uploaded_file is not None:
            streamlit.write("已成功上傳檔案:", uploaded_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
         
            streamlit.subheader('表單结果')
            streamlit.write(main(user_input, temp_file.name)) 