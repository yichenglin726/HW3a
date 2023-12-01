#reference 
#https://discuss.streamlit.io/t/problem-in-reading-a-db-object-using-file-uploader/3064/6
#https://discuss.streamlit.io/t/st-form-getting-displayed-before-st-title/40768/6
import camelot
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        fp = "temporary_for_hw.pdf"

        #write all the values of the original pdf to temporary file
        try:
            f_open = open(fp, "wb")
            f_open.write(pdf_file.getvalue())
        finally:
            f_open.close()

            #read the temporary file
            tables = camelot.read_pdf(fp, pages="all")
            texts = []
            tables_df = []
            for table in tables:
                texts.append(table.df.to_string())
                tables_df.append(table.df)

        os.remove(fp)
        return texts, tables_df


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-xlm-r-multilingual-v1')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_table, vector_from_keyword)

def search_table(uploaded_files, search_query):
    if not uploaded_files:
        st.warning("You forgot to upload the pdf file")
    elif not search_query:
        st.warning("Search query is still empty")
    else:
        pdf_parser = pdf2text() 
        text_vector = text2vector()
        cos_sim = cosine_sim()

        max_cos_sim = 0
        result = None
        search_query_vector = text_vector(search_query)

        for file in uploaded_files:
            texts, tables_df = pdf_parser(file)
            len_list = len(texts)

            for i in range(0, len_list):
                cur_text_vector = text_vector(texts[i])
                cur_cos_sim = cos_sim(cur_text_vector, search_query_vector).item()
                if cur_cos_sim > max_cos_sim:
                    max_cos_sim = cur_cos_sim
                    result = tables_df[i]
        
        st.session_state.table_result = result

def main():
    #streamlit for UI
    st.title("Big Data Systems Homework 3a")
    st.subheader("郭雅美 B09902085")

    uploaded_files = st.file_uploader("Choose your pdf files", type=["pdf"], accept_multiple_files=True)
    search_query = st.text_input("Enter your search query")
    search_button = st.button("Press to Search", on_click= search_table, args= (uploaded_files, search_query))

    if search_button and "table_result" in st.session_state:
        st.dataframe(st.session_state.table_result)

if __name__ == "__main__":
    main()
