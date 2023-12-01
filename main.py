import camelot
from sentence_transformers import SentenceTransformer, models, util
import numpy as np
import streamlit as st
from typing import List
from glob import glob


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        table_df = []
        for table in tables:
            # print(ta)
            texts.append(table.df.to_string())
            table_df.append(table.df.replace('\n', '', regex=True).replace("\\n", "", regex=True).replace(" ", "", regex=True))
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text, table_df


class text2vector:
    def __init__(self):
        word_embedding_model = models.Transformer('hfl/chinese-roberta-wwm-ext')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        # self.model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    def __call__(self, text):
        vector = self.model.encode(text)

        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.pytorch_cos_sim(vector_from_table, vector_from_keyword).item()


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text, table_df = pdf_parser(pdf_file)
    # for table in table_df:
    #     print(table)
    #     print("====================================")
    text_parser = text2vector()
    similarity = cosine_sim()
    embedding = []
    similarity_socre = []
    for table in table_df:
        input_table = table.to_string()
        embedding.append(text_parser(input_table))
    
    keyword_embedding = text_parser(keyword)
    for e in embedding:
        similarity_socre.append(similarity(e, keyword_embedding))
    max_index = np.argmax(similarity_socre)
    print(similarity_socre)
    return table_df[max_index]


if __name__ == "__main__":
    # table = main("監督式學習的優缺點", "docs/1.pdf")
    # print(table)
    # table = main("動物細胞特點", "docs/2.pdf")
    # print(table)

    st.set_page_config(page_title="PDF Table Search", layout="wide")
    st.title("Document Intelligence - PDF Table Search")

    pdf_info = {
        "🤖 1.pdf": "監督式學習、非監督式學習、強化學習相關的資料表格",
        "🔬 2.pdf": "動植物細胞特點、多細胞生物和單細胞生物相關的資料表格",
    }
    for pdf, intro in pdf_info.items():
        st.write(f"{pdf}: {intro}")

    pdf_files = [f for f in glob("docs/*.pdf")]
    selected_pdf = st.multiselect("Select PDF files", pdf_files, default=pdf_files)

    keyword = st.text_input("Enter a keyword")

    if selected_pdf and keyword:
        with st.spinner("🔎 Searching.."):
            table = main(keyword, selected_pdf)
        st.header("Search Result")
        st.subheader(table["title"])
        st.dataframe(table["table_df"])