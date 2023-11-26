import camelot
import math
import re
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        titles = self.get_titles(pdf_file)
        tables_data = []

        for table, title in zip(tables, titles):
            table.df.replace(r'\n', '', regex=True, inplace=True)
            tables_data.append({
                "title": title,
                "table_df": table.df,
                "table_string": table.df.to_string().replace("\\n", ""),
            })

        return tables_data

    def get_titles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
        titles = []
        
        for table in tables:
            text = table.df.to_string().replace("\\n", "").strip().split(' ')
            for word in text:
                if "ai_tables" in word:
                    titles.append(word+" "+text[text.index(word)+1])

        return titles


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        return self.model.encode(text)
        


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword).item()


def get_table(keyword, pdf_file):
    pdf_parser = pdf2text()
    t2v = text2vector()
    sim = cosine_sim()
    tables_df = pdf_parser(pdf_file)
    keyword_vect = t2v(keyword)
    sim_val = []

    #Calculate similarity of all tables
    for table_df in tables_df:
        table_vect = t2v(table_df["table_string"])
        similarity = sim(table_vect, keyword_vect)
        sim_val.append(similarity)

    #Find target index
    target_index = sim_val.index(max(sim_val))
    target_table = tables_df[target_index]["table_df"]
    target_title = tables_df[target_index]["title"]
    return target_title, target_table

def main():
    #UI
    st.set_page_config(
        page_title="Education Table Search AI",
        page_icon="🧙🏽‍♂️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.title("Education Table Search AI🧙🏽‍♂️")
    #PDF Title
    st.write("PDF 1: 機器學習")
    st.write("PDF 2: 動植物細胞")
    st.subheader("📝Select a PDF file: ")
    #Add more pdf here
    pdf_files = {1: "docs/1.pdf", 2: "docs/2.pdf"}
    pdf_names = {1: "機器學習", 2: "動植物細胞"}

    def format_func(option):
        return pdf_names[option]

    option = st.selectbox("", options=list(pdf_names.keys()), format_func=format_func)
    pdf_file = pdf_files[option]

    st.subheader("✍🏿Input a keyword: ")
    keyword = st.text_input("")
    serch = st.button("🔎Search")

    if serch or (pdf_file and keyword):
        with st.spinner('Searching🔄🔄🔄'):
            title, table = get_table(keyword, pdf_file)
            st.subheader(title)
            st.dataframe(table)





if __name__ == "__main__":
    main()
