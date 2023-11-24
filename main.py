import camelot
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import streamlit as st
import os

class df2pairs:
    def __init__(self):
        pass

    def __call__(self, df_table):
        non_numeric_row = None
        non_numeric_column = None
        pair_content = []

        for index, row in df_table.iterrows():
            for col in df_table.columns:
                if not str(row[col]).isdigit() and str(row[col]) != '':
                    non_numeric_row = index
                    non_numeric_column = col
                    break
            if non_numeric_row is not None:
                break
        for row in range((non_numeric_row+1),(df_table.shape[0]-non_numeric_row)):
            for col in range((non_numeric_column+1),(df_table.shape[1]-non_numeric_column)):
                pair_content.append(f'{df_table.iloc[non_numeric_row, col]}的{df_table.iloc[row, non_numeric_column]}:{df_table.iloc[row, col]}')

        return pair_content
class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        df_list = []
        texts = []
        pairs = []
        for table in tables:
            texts.append(table.df.to_string())
            df_list.append(table.df.replace('\n', '', regex=True).replace("\\n", "", regex=True).replace(" ", "", regex=True))
            pairs.append(df2pairs()(table.df))
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text, df_list, pairs
class text2vector_cosinesim:
    def __init__(self):
        # Load model directly
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1', cache_folder="./")

    def __call__(self, vector_from_table, vector_from_keyword):
        vector1 = self.model.encode(vector_from_table)
        vector2 = self.model.encode(vector_from_keyword)
        cosine = np.dot(vector1,vector2)/(norm(vector1)*norm(vector2))
        return cosine
class getmaxsim:
    def __init__(self):
        pass

    def __call__(self, content_list, keyword):
        max_sim_idx = 0
        max_sim = 0
        for i in range(len(content_list)):
            for j in range(len(content_list[i])):
                sim = text2vector_cosinesim()(content_list[i][j], keyword)
                # print(content_list[i][j],sim)
                if sim > max_sim:
                    max_sim = sim
                    max_sim_idx = i
                    
        return max_sim_idx
def main(keyword, pdf_file):
 
    pdf_parser = pdf2text()
    _, table_lists, content_pairs = pdf_parser(pdf_file)
    target_id = getmaxsim()(content_pairs, keyword)

    return table_lists[target_id]

if __name__ == "__main__":
    st.title('BDS HW3a - 張庭維')
    st.write("Please upload your pdf file and input your keyword")
    upload_file = st.file_uploader("Upload YOUR PDF File", type=['pdf'])
    if upload_file is not None:
        st.write(f"Your file is {upload_file.name}")
        pdf_file = os.path.join("docs", upload_file.name)
        with open(pdf_file, 'wb') as f:
            f.write(upload_file.read())
    
    st.write("Type your keyword")
    keyword = st.text_input("keyword", "非監督式學習的應用")
    
    if st.button("Search"):
        table = main(keyword, pdf_file)
        st.write("Result:")
        st.write(table)