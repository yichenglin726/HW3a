import camelot
import streamlit as st
import time
from scipy import spatial
from text2vec import SentenceModel
import numpy as np
from tempfile import NamedTemporaryFile

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
            f.write(pdf_file.getbuffer())
            tables = camelot.read_pdf(f.name, pages="all")
        
        texts = []
        for table in tables:
            texts.append(table.df)
        
        return texts


class table2sentence:
    def __call__(self, table):
        num_of_cell = len(table.split()) + 1
        num_of_row = table.count('\n') + 1
        num_of_column = (int)(num_of_cell / num_of_row)
        table_text_vector = [i.lower() for i in table.split()]
        
        table_text_vector = table_text_vector[num_of_column-1:len(table_text_vector)]
        
        mask = np.array(range(len(table_text_vector)))
        mask = (mask % num_of_column) != 0

        table_text_vector = np.array(table_text_vector)
        table_text_vector = table_text_vector[mask]

        
        # #print(table_text_vector)

        # num_of_row = num_of_row - 1
        # num_of_column = num_of_column - 1
        # #print(num_of_column, num_of_row, len(table_text_vector))

        # columns = [table_text_vector[i::num_of_column] for i in range(num_of_column)]
        # rows = [table_text_vector[i:i+num_of_column] for i in range(0, len(table_text_vector), num_of_column)]

        # #print(columns)
        # #print(rows)

        # sentences = []
        
        # padding1 = np.repeat('為', num_of_row)
        # padding2 = np.repeat('，', num_of_row)
        # padding2[-1] = '。'
        # for i in range(1, len(columns)):
        #     words = np.vstack((columns[0], padding1[:len(columns[0])], columns[i], padding2[:len(columns[i])])).reshape((-1,),order='F')
        #     setence = ''.join(words)
        #     #print(setence)
        #     sentences.append(setence)

        # padding1 = np.repeat('為', num_of_column)
        # padding2 = np.repeat('，', num_of_column)
        # padding2[-1] = '。'
        # for i in range(1, len(rows)):
        #     words = np.vstack((rows[0], padding1[:len(rows[0])], rows[i], padding2[:len(rows[i])])).reshape((-1,),order='F')
        #     setence = ''.join(words)
        #     #print(setence)
        #     sentences.append(setence)
        
        # sentences.append(''.join(table_text_vector))
        # print(sentences)

        # return sentences
        
        table_text_vector = '，'.join(table_text_vector)

        return table_text_vector


class text2vector:
    def __init__(self):
        self.model = SentenceModel("shibing624/text2vec-base-multilingual")

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return 1 - spatial.distance.cosine(vector_from_table, vector_from_keyword)


def find_best_table(keyword, pdf_files):
    pdf_parser = pdf2text()
    table_parser = table2sentence()
    text_tranformer = text2vector()
    get_cosine_sim = cosine_sim()

    best_cosine_sim = -1
    best_table = None
    keyword_vector = text_tranformer(keyword)

    for pdf_file in pdf_files:

        tables = pdf_parser(pdf_file)
        
        for table in tables:
            table_vector = text_tranformer(table_parser(table.to_string()))
            sim = get_cosine_sim(table_vector, keyword_vector)
            
            if sim > best_cosine_sim:
                best_cosine_sim = sim
                best_table = table
        
    return best_table

class UI:
    def show(self):
        if 'step2' not in st.session_state:
            st.session_state["step2"] = True
        if 'step3' not in st.session_state:
            st.session_state["step3"] = True

        st.title('CSIE 5322: Big Data System')
        st.header('HW3: Stage-A Document Intelligence')
        st.subheader('資工所碩一 R12922076 鄭仲語', divider='rainbow')

        st.header('Searches in which table in the given pdf files has the desired information!')
        
        st.subheader('Step 1:')
        pdfs = st.file_uploader('Upload PDF files with only tables inside.', \
                                type = ['pdf'], \
                                    help='Only pdf files are allowed.',\
                                        on_change=self.on_file_change,\
                                            accept_multiple_files=True,\
                                                key='my_file_uploader')

        st.subheader('Step 2:')
        keyword = st.text_input('Type searching keywords for tables in the above pdf files.',\
                                help='This field cannot be empty.',\
                                    on_change=self.on_text_change,\
                                        disabled=st.session_state.step2,\
                                            placeholder='Example: 非監督式學習的應用',\
                                                key='my_text_input')

        st.subheader('Step 3:')
        st.write('Click the button for searching!')

        st.button('Search the keyword!',\
                  type='primary',\
                    disabled=st.session_state.step3,\
                        on_click=self.on_click, args=(pdfs, keyword))
        
        st.divider()

        if 'table' in st.session_state:
            st.table(st.session_state.table)

    def on_file_change(self):
        st.session_state.step2 = len(st.session_state.my_file_uploader) == 0
        placeholder = st.empty()
        placeholder.success('File changes success!', icon="✅")
        time.sleep(1)
        placeholder.empty()

    def on_text_change(self):
        st.session_state.step3 = len(st.session_state.my_file_uploader) == 0 \
            and len(st.session_state.my_text_input) == 0
        placeholder = st.empty()
        placeholder.success('Text input success!', icon="✅")
        time.sleep(0.5)
        placeholder.empty()

    def on_click(self, pdfs, keyword):
        table = find_best_table(keyword, pdfs)
        st.session_state.table = table

if __name__ == "__main__":
    ui = UI()
    ui.show()