#!/usr/bin/env python
# coding: utf-8

# In[105]:


import camelot
import pdfplumber
import numpy as np
import re
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ipywidgets as widgets
from IPython.display import display


# In[106]:


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        titles = []
        table_text = []
    
        with pdfplumber.open(pdf_file) as pdf:
            draft = ""
            for page in pdf.pages:
                if page.extract_text():
                    draft += page.extract_text() + "\n"

            titles = re.findall(r'ai_tables_#.*', draft)
    
        for table in tables:
            #table_text.append(table.df)
            table_text.append((table.df.to_string()).replace("\\n", ""))
            titles.append(table.df.to_string().replace("\\n", ""))
        return titles, table_text


# In[107]:


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)


# In[108]:


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine = np.dot(vector_from_table,vector_from_keyword)/(norm(vector_from_table)*norm(vector_from_keyword))
        return cosine


# In[109]:


def main(keyword, uploaded_file):
    pdf2txt = pdf2text()
    text2vec = text2vector()
    calc_cossim = cosine_sim()
    titles, table_texts = pdf2txt(uploaded_file)
    #print(table_texts)
    
    tablevectors = []
    for i in table_texts:
        tablevectors.append(text2vec(i))

    tablecosine = []
    for table_vector in tablevectors:
        tablecosine.append(calc_cossim(table_vector, text2vec(keyword)))
    
    print("\n")
    print("Result:")
    max_index = tablecosine.index(max(tablecosine))
    print(titles[max_index])
    print(table_texts[max_index])

if __name__ == "__main__":
# main("非監督式學習的應用", "C:\\Users\\H P\\Downloads\\data1.pdf")
#     uploaded_file = input("Input your pdf datapath: ")
#     keyword = input("Input your keyword: ")
#     main(keyword, uploaded_file)
    print("BDS HW3 - 黃麗企")
    
    file_input = widgets.Text(description='File Path:')
    keyword_input = widgets.Text(description='Keyword:')
    button = widgets.Button(description='Search')
    
    def on_button_clicked(b):
        uploaded_file = file_input.value
        keyword = keyword_input.value
        main(keyword, uploaded_file)
        
    button.on_click(on_button_clicked)
    display(widgets.VBox([file_input, keyword_input, button]))


# In[ ]:




