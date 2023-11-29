import camelot
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file): 
        with open('docs/temp.pdf', "wb") as f:
            f.write(pdf_file.getvalue())
        tables = camelot.read_pdf('docs/temp.pdf', pages="all")
        texts = []
        df = []
        for table in tables:
            df.append(table.df)
            # texts.append(table.df.to_string())
            text = table.df.to_string()
            # text = "\n".join(texts)
            text = text.replace("\\n", "")
            texts.append(text)
        return texts, df


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('shibing624/text2vec-base-chinese')

    def __call__(self, text):
        embedding = self.model.encode(text, convert_to_tensor=False)
        # print(embedding)
        return embedding


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        cosine_scores = util.cos_sim(vector_from_table, vector_from_keyword)
        return cosine_scores[0][0]


def main(keyword, pdf_files):
    pdf_parser = pdf2text()
    vector_parser = text2vector()
    cosine_parser = cosine_sim()

    max_sim = -1
    max_sim_df = None
    for pdf_file in enumerate(pdf_files):
        table_text, table_df = pdf_parser(pdf_file[1])
        table_vec = vector_parser(table_text)
        keyword_vec = vector_parser(keyword)

        for i, t_vec in enumerate(table_vec):
            sim = cosine_parser(t_vec, keyword_vec)
            if sim > max_sim:
                max_sim = sim
                max_sim_df = table_df[i]
                # print(max_sim_df)
    # print(table_df[max_sim_id])
    return max_sim_df

class UI:
    def show(self):
        st.title("BDS HW3a")
        st.write("R11922101 Chia-Hung Huang")
        st.write("Create a program that searches in which table in the given pdf files has the desired information.")

        pdfs = st.file_uploader("Upload pdf files with only tables inside", type=['pdf'], accept_multiple_files=True)
        keyword = st.text_input("The search keywords")

        if st.button("Submit", type="primary"):
            if pdfs == []:
                st.error("PDF files are required.")
                return
            elif keyword == "":
                st.error("The search keywords are required.")
                return
            else:
                with st.spinner('Processing...'):
                    table = main(keyword, pdfs)
                    st.write("Result table:")
                    st.write(table)

if __name__ == "__main__":
    # main("keyword", "docs/1.pdf")
    # main("keyword", "docs/2.pdf")
    ui = UI()
    ui.show()
