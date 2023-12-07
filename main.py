import camelot 
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
import streamlit as st

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        text = "\n".join(texts)
        text = text.replace("\\n", "")
        return text


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return util.cos_sim(vector_from_keyword, vector_from_table)


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)
    print(table_text)
    # return table


if __name__ == "__main__":
    main("keyword", "docs/1.pdf")
    main("keyword", "docs/2.pdf")
