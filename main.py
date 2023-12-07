import camelot, os, math
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util


class pdf2text:
    def __init__(self):
        pass

    def getTables(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            # 刪除空字串的row
            table.df = table.df[~table.df.map(lambda x: x.strip() == "").all(axis=1)]
            max_len = max([len(t) for t in table.df.iloc[0] if isinstance(t, str)])

            if max_len > 10:
                texts[-1] = pd.concat([texts[-1], table.df], axis=0, ignore_index=True)
            else:
                texts.append(table.df)

        return texts

    def getTitles(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, flavor="stream", pages="all")
        titles = []
        for table in tables:
            text = table.df.to_string(index=False).split("\n")
            text = [x.replace(" ", "") for x in text if x != ""]
            for word in text:
                if "tables" in word:
                    titles.append(word)

        for title in titles:
            if titles.count(title) > 1:
                titles.remove(title)

        return titles


class text2vector:
    def __init__(self):
        # Load the SentenceTransformer model
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    def __call__(self, text):
        # Convert the input text to a vector
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, table_vec, key_vec):
        """Calculate cosine similarity between 2 vectors

        Args:
        - table_vec: token vector from the input table
        - key_vec: keyword to be compared
        """
        return util.cos_sim(table_vec, key_vec)


def best_idx(pred_tab, pred_title):
    table_s = max(pred_tab)
    title_s = max(pred_title)
    print(f"tab: {pred_tab}")
    print(f"title: {pred_title}")
    if title_s > table_s and title_s > 0.1:
        print("by title")
        return pred_title.index(title_s), title_s
    else:
        print("by table")
        return pred_tab.index(table_s), table_s


def Searching(keyword, pdf_file):
    if pdf_file == "" or os.path.getsize(pdf_file) == 0:
        print("PDF file is empty.")
        return

    pdf_parser = pdf2text()
    text2vec = text2vector()
    cos_sim = cosine_sim()
    tables = pdf_parser.getTables(pdf_file)
    titles = pdf_parser.getTitles(pdf_file)

    key_vec = text2vec(keyword)

    pred_tab, pred_title = [], []
    for data in zip(tables, titles):
        table, title = data
        table_vector = text2vec(table.to_string(index=False))
        title_vector = text2vec(title)

        similarity_table = cos_sim(table_vector, key_vec)
        similarity_title = cos_sim(title_vector, key_vec)
        pred_tab.append(similarity_table)
        pred_title.append(similarity_title)

    max_id, score = best_idx(pred_tab, pred_title)

    # return pred table
    return tables[max_id], titles[max_id], score


class PDFDocumentIntelligenceApp:
    def __init__(self):
        file_names = os.listdir("./docs")
        self.filename_list = []
        for file_name in file_names:
            self.filename_list.append(file_name)
        self.filename_list.sort()

        ##----UI----##
        self.title = st.title("PDF Document AI by r12921a10")
        self.exist_pdf = st.selectbox("Choose an exist file", self.filename_list)
        self.upload_pdf = st.file_uploader("Upload PDF file", type=["pdf"])
        self.keyword = st.text_input("Input keywords")
        self.output_text = st.text("")
        self.pdf_file_path = ""
        ##

        if not self.exist_pdf.endswith(".pdf"):
            self.exist_pdf = self.exist_pdf + ".pdf"

        if self.exist_pdf in self.filename_list:
            self.pdf_file_path = os.path.join("docs", self.exist_pdf)
        elif self.upload_pdf:
            self.pdf_file_path = os.path.join("docs", self.upload_pdf.name)
            with open(self.pdf_file_path, "wb") as f:
                f.write(self.upload_pdf.read())
                self.filename_list.append(file_name)

    def process_pdf(self):
        if self.keyword != "":
            table, title, score = Searching(self.keyword, self.pdf_file_path)
            st.text(f"score: {score.item():.3f}")
            st.text(title)
            st.dataframe(table)


if __name__ == "__main__":
    app = PDFDocumentIntelligenceApp()
    output = app.process_pdf()
