# %%
import os
os.environ['TORCH_HOME'] = '.cache'

import camelot
from sentence_transformers import SentenceTransformer, util
import torch

class PdfParser:
    def __init__(self):
        pass

    def extract_text_from_pdf(self, pdf_file):
        pdf = PdfFileReader(pdf_file)
        text = ""
        for page in range(pdf.getNumPages()):
            text += pdf.getPage(page).extractText()
        return text

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        tables_df = [table.df for table in tables]
        all_texts = []
        for table in tables_df:
            texts = []
            texts.append(table.to_string())
            text = "\n".join(texts)
            text = text.replace("\\n", "")
            all_texts.append(text)
        return tables_df, all_texts
# %%
def search(keyword, pdf_file):
    # %%

    # debug
    # keyword = "監督學習的應用"
    # pdf_file = "docs/1.pdf"

    pdf_parser = PdfParser()
    print(f"** Parsing PDF file {pdf_file} **")
    tables, table_text = pdf_parser(pdf_file)
    print("** Done! **")

    # %%
    print("** Encoding files and keywords **")
    embedder = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    table_vector = embedder.encode(table_text)
    keyword_vector = embedder.encode(keyword)
    print("** Done! **")

    # search for the most similar table
    print("** Searching for the most similar table **")
    cos_scores = util.cos_sim(table_vector, keyword_vector)
    print("** Done! **")

    print("** Results **")
    max_table = tables[torch.argmax(cos_scores)]
    print("Keyword: ", keyword)
    print("Most similar table: ")   
    print(max_table)

    # %%
    return max_table

    # %%