import camelot
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import streamlit as st
import os
class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        df = []
        for table in tables:
            df.append(table.df)
        return df


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer("shibing624/text2vec-base-chinese").cuda()
        
    def __call__(self, text):
        return self.model.encode(text,convert_to_tensor=True)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        similarity = F.cosine_similarity(vector_from_table, vector_from_keyword,dim=0)
        return similarity.item()  


def search(keyword,chosen_pdf):
    pdf2text_ = pdf2text()
    text2vector_ = text2vector()
    cosine_sim_ = cosine_sim()
    max_score = 0
    df_dict = {file:pdf2text_(file) for file in chosen_pdf}
    for file in df_dict:
        for df in df_dict[file]:
            score = cosine_sim_(text2vector_(df.to_string().replace('\n', '').replace('\\n', '')), text2vector_(keyword))
            # features = []
            # scores = []
            # for row in df.values: features += list(map(lambda x: x.replace('\n', '').replace('\\n', ''),row))
            # for feature in features:
            #     if feature:
            #         feature_score = cosine_sim_(text2vector_(feature), text2vector_(keyword))
            #         if feature_score > 0.6: 
            #             scores.append(feature_score)
            # if not scores: continue
            # score = sum(scores)/len(scores)
            if score > max_score: 
                max_score = score
                output = [file,df,max_score]

    return output




if __name__ == "__main__":
    docs_path = './docs/'
    pdf_files = os.listdir(docs_path)
    st.title("Table Searching Tool for PDF Files")
    chosen_pdf = st.multiselect("Choose PDF files to search for tables", pdf_files)
    st.write(f"Chosen PDF files: {', '.join(chosen_pdf)}")
    keyword = st.text_input("Please enter a keyword")
    if st.button("Search"):
        with st.spinner("Searching the most related table..."):
            file,table,score = search(keyword, [docs_path + file for file in chosen_pdf])
            st.subheader("Result")
            st.dataframe(table)
            st.write(f"Score: {score:4f}, Src file: {file.split('/')[-1]}")
