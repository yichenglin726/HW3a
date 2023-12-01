import os
import re
import sys
import io
import camelot
import math
import torch
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from camelot.handlers import PDFHandler
from text2vec import SentenceModel
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

def pdf2text(pdf_file): # relative path to the file
    # get table title
    tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
    titles = list()
    for table in tables:
        text = table.df.to_string().replace("\\n", "").strip().split(' ')
        for word in text:
            if "table" in word:
                titles.append(word+" "+text[text.index(word)+1])
    titles = sorted(titles)

    # get table content and add title
    title_number = 0
    tables = camelot.read_pdf(pdf_file, pages='all', split_text=True)
    texts = list()
    for table in tables:
        # texts.append([[string.replace('\n', "") for string in nested] for nested in table.df.values.tolist()])
        texts.append(titles[title_number] + "\n" + table.df.to_string().replace("\\n", ""))
        title_number += 1
    return texts

def text2vector(text):
    model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    embeddings = model.encode(text)
    return embeddings

def cosine_sim(vector, key_vector):
    return F.cosine_similarity(vector, key_vector, dim=0)


def main(file_dir, queries):
    # find all files in the path
    pdf_paths = list()
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            pdf_paths.append(os.path.join(root,file))
            #print(file_path), print(os.path.abspath(file_path))
    
    # process
    query_number = 0
    sim_table = list()
    for pdf in pdf_paths:
        text = pdf2text(pdf)
        print("===pdf2text is done===")
        vector = torch.tensor(text2vector(text))
        key_vector = torch.transpose(torch.tensor(text2vector(queries[query_number])), 0, 1)
        print(vector.shape)
        print(key_vector.shape)
        print("===text2vector is done===")
        query_number += 1
        # # calculate similarity
        for i in range(vector.shape[0]):
            temp = cosine_sim(vector[i], key_vector)
            print(temp.shape)
            print("")
            sim_table.append(temp)
        # result = cosine_sim(vector, key_vector)
        print("===cosine_sim is done")
        print(sim_table)

    pass


if __name__ == "__main__":
    file_dir = "./docs"
    queries = [["非監督式學習的應用"], ["多細胞生物細胞膜和植物細胞膜的比較"]]
    main(file_dir, queries)