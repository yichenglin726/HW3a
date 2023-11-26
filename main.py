import camelot
from sentence_transformers import SentenceTransformer, util
import numpy as np
from numpy.linalg import norm
import streamlit as st
import os


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts, dfs = [table.df.to_string().replace("\\n", "") for table in tables], [table.df for table in tables]
        return texts, dfs

class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text, convert_to_tensor=False)


class CosineSimilarity:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return np.dot(vector_from_table,vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))

def UI(custom_keyword):
    st.title('BDS HW3a - PDF2Table')
    upload_file = st.file_uploader("Upload Your PDF File", type=['pdf'])
    if upload_file is not None:
        st.write("Selected file : ", upload_file.name)
        pdf_file = os.path.join("docs", upload_file.name)
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())
    
    # keyword = st.text_input("keyword", "非監督式學習") 
    # keyword = st.text_input("keyword", "細胞膜") 
    keyword = st.text_input("keyword", custom_keyword) 
    
    if st.button("Search"):
        table = main(keyword, pdf_file)
        st.write("Output:", table)
        
def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text_parser = text2vector()
    cos_sim_calulator = CosineSimilarity()
    
    table_texts, table_dfs = pdf_parser(pdf_file)
    # table_vector = text_parser(table_texts)
    table_vectors = [text_parser(text) for text in table_texts]
    keyword_vector = text_parser(keyword)
    
    cos_sims = [cos_sim_calulator(vec, keyword_vector) for vec in table_vectors]
    return table_dfs[np.argmax(cos_sims)]

if __name__ == "__main__":
    # UI()
    UI("Enter Keyword Here")