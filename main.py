import camelot
from sentence_transformers import SentenceTransformer, util
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
        table_df = []

        for table in tables:
            if table.df.index[-1] <= 2: # adjacent
                temp_table = table.df

            else:
                if temp_table is not None:
                    current_table = pd.concat([temp_table, table.df], ignore_index=True)
                    temp_table = None
                else:
                    current_table = table.df

            table_df.append(current_table)
            text = current_table.to_string()
            text = text.replace("\\n", "")
            texts.append(text)

        titles = self.get_titles(pdf_file)
        return texts, table_df, titles

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
        if chosed_pdf:
            st.subheader("Chosed File")
            st.write(chosed_pdf.name)
            pdf_file = os.path.join("docs", chosed_pdf.name)
            with open(pdf_file, 'wb') as f:
                f.write(chosed_pdf.read())
            st.subheader("Please Enter Keywords")
            keyword = st.text_input("keyword", "Enter Keywords Here")
    
            if st.button("Search"):
                with st.spinner('Searching...'):
                    table, title = main(keyword, pdf_file)
                    st.write("Result")
                    st.write(title)
                    st.dataframe(table)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    text2vector_parser = text2vector()
    cosine_sim_parser = cosine_sim()

    table_text, table_df, titles = pdf_parser(pdf_file)
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
    title = titles[max_index]

    return table, title


if __name__ == "__main__":
    GUI()