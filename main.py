import camelot
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def pdf2text(pdf_file): # relative path to the file
    # get table title
    tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
    titles = list()
    for table in tables:
        text = table.df.to_string().replace("\\n", "").strip().split(' ')
        for word in text:
            if "table" in word: titles.append(word+" "+text[text.index(word)+1])
    titles = sorted(titles)
    # get table content and add title
    title_number = 0
    tables = camelot.read_pdf(pdf_file, pages='all', split_text=True)
    texts = list()
    for table in tables:
        texts.append(titles[title_number] + "\n" + table.df.to_string().replace("\\n", ""))
        title_number += 1
    return texts

def text2vector(text):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    return model.encode(text)

def cosine_sim(vector, key_vector):
    return F.cosine_similarity(vector, key_vector, dim=0)


def search(file_dir, file_index, query):
    pdf_paths = list()
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            pdf_paths.append(os.path.join(root,file))

    sim_table = list()
    text = pdf2text(pdf_paths[file_index])
    vector = torch.tensor(text2vector(text))
    key_vector = torch.transpose(torch.tensor(text2vector([str(query)])), 0, 1)

    sim_table = list()
    for i in range(vector.shape[0]):
        sim_table.append(cosine_sim(vector[i], key_vector).argmax())
    result_idx = sim_table.index(max(sim_table))

    return text[result_idx]

def main(file_dir):
    st.title("AI Table Search Ingine")
    st.write("😎Input file: 1.pdf&nbsp;&nbsp;&nbsp;&nbsp;🎓Query: 非監督式學習的應用")
    st.write("😎Input file: 2.pdf&nbsp;&nbsp;&nbsp;&nbsp;🎓Query: 多細胞生物細胞膜和植物細胞膜的比較")
    st.subheader("Select a PDF file")
    file_index = int(Path(st.selectbox("PDF file", ["./docs/1.pdf", "./docs/2.pdf"])).stem)-1
    st.subheader("Input a query keyword here")
    keyword = str(st.text_input("keyword"))
    submit_btn = st.button("search")
    if submit_btn:
        with st.spinner('Wait a minutes, the model is searching the result...'):
            result = search(file_dir, file_index, keyword)
            st.write(result)

if __name__ == "__main__":
    main("./docs")