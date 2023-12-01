import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import streamlit as st

class pdf2text:
    def __init__(self):
        pass
    def __call__(self, path):
        pdf = pdfplumber.open(path)
        table = pdf.pages[0].extract_tables()
        for i in range(len(table)):
            for j in range(len(table[i])):
                table[i][j] = [item.replace('\n', '') for item in table[i][j]]
        return table
    
class text2vector():
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, input):
        vector = self.model.encode(input)
        return vector

class cos_sim:
    def __init__(self):
        pass
    def __call__(self, table, key_word):
        v_table, v_word = text2vector().__call__(table), text2vector().__call__(key_word)
        sim = np.dot(v_table, v_word) / (np.linalg.norm(v_table) * np.linalg.norm(v_word))
        return sim
    
class cr_pair_sim:
    def __init__(self):
        pass
    def __call__(self, table, key_word):
        sim = []
        max_sim = 0
        max_sim_table = -1
        for i in range(len(table)):
            for col in range(len(table[i][0])):
                for row in range(len(table[i])):
                    sim.append(cos_sim().__call__(table[i][0][col]+'的'+table[i][row][0], key_word))
                    if sim[-1] > max_sim:
                        max_sim = sim[-1]
                        max_sim_table = i
        return table[max_sim_table]

def main(pdf, key_word):
    table = pdf2text().__call__(pdf)
    max_sim_table = cr_pair_sim().__call__(table, key_word) 
    table = pd.DataFrame(max_sim_table[1:], columns=max_sim_table[0])
    return table

if __name__ == "__main__":
    st.title('BDS HW3: pdf table search')
    st.write('Author : 王睿誼')
    st.header('Choose a topic :')
    st.write('Topic 1 : ML related.')
    st.write('Topic 2 : Cell related.')
    select = st.selectbox('Please select a topic', ['Topic 1', 'Topic 2'])
    st.header('Input a keyword :')
    keyword = st.text_input('keyword', '非監督式學習的應用')
    if st.button('Start searching'):
        if select == 'Topic 1':
            pdf = 'BDS/HW3a-main/docs/1.pdf'
        else:
            pdf = 'BDS/HW3a-main/docs/2.pdf'
        st.header('The result is :')
        table = main(pdf, keyword)
        st.table(table)
    