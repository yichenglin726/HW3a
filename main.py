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
        #pass
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        #pass
        return self.model.encode(text, convert_to_tensor=False)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        #pass
        cos = np.dot(vector_from_table,vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))
        return cos

#def format_func(file_list):
#    if file_list != []:
#        return [s for s in file_list if "name" in s]

def GUI():
    st.title('Big Data System HW3a')
    st.header("PDF Table Search")
    st.write("Copyright © 2023 K.T. Tu. All Rights Reserved.")
    st.subheader("Upload Your PDF File")
    submit_file = st.file_uploader("Upload PDF File", type=['pdf'], accept_multiple_files=True)
    if submit_file != []:
        chosed_pdf = st.selectbox("Choose PDF File", submit_file)
        #chosed_pdf = st.selectbox("Choose PDF File", submit_file, format_func=format_func(submit_file))
        if chosed_pdf:
            st.subheader("Chosed File")
            st.write(chosed_pdf.name)
            pdf_file = os.path.join("docs", chosed_pdf.name)
            with open(pdf_file, 'wb') as f:
                f.write(chosed_pdf.read())
            st.subheader("Please Enter Keywords")
            keyword = st.text_input("keyword", "Enter Keywords")
    
            if st.button("Search"):
                table = main(keyword, pdf_file)
                st.write("Result")
                st.write(table)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    table_text, table_df = pdf_parser(pdf_file)
    #print(table_text)

    text_vector = text2vector_parser(table_text)
    keyword_vector = text2vector_parser(keyword)
    
    cos_sim = []
    for i in range(len(text_vector)):
        cos_sim.append(cosine_sim_parser(text_vector[i], keyword_vector))
    
    # find the max cosine similarity
    max_cos_sim = max(cos_sim)
    max_index = cos_sim.index(max_cos_sim)
    table = table_df[max_index]

    return table


if __name__ == "__main__":
    GUI()