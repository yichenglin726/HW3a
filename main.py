# %%
import camelot
from sentence_transformers import SentenceTransformer, util
import torch
# %%
class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        table_dfs = [table.df for table in tables]
        table_texts = []
        for table in tables:
            table_text = [table.df.to_string()]
            table_text = "\n".join(table_text)
            table_text = table_text.replace("\\n", "")
            table_texts.append(table_text)
        return table_dfs, table_texts


class text2vector:
    def __init__(self):
        # using pretrained model
        self.model = SentenceTransformer("all-mpnet-base-v2")

    def __call__(self, text):
        # encode text
        vector = self.model.encode(text)
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        # compute cosine similarity
        cos_sim = util.pytorch_cos_sim(vector_from_table, vector_from_keyword)
        return cos_sim
# %%
class SearchEngine:
    def __init__(self):
        self.embedding = text2vector()
        self.cosine_sim = cosine_sim()
        self.pdf_parser = pdf2text()
        
    def search(self, keyword, pdf_file):
        # from pdf to text
        tables, table_texts = self.pdf_parser(pdf_file)
        
        # from text to vector
        embedding_table = self.embedding(table_texts)
        embedding_keyword = self.embedding(keyword)
        
        # compute cosine similarity
        cos_sim = self.cosine_sim (embedding_table, embedding_keyword)
        # the most similar table
        max_cos_sim_index = torch.argmax(cos_sim)
        return tables[max_cos_sim_index], cos_sim[max_cos_sim_index]
        # return table

def main(keyword, pdf_file):
    search_engine = SearchEngine()
    table, score = search_engine.search(keyword, pdf_file)
    return table, score

if __name__ == "__main__":
    table, score = main("非監督式學習的應用", "docs/1.pdf")
    print("Search keyword: 非監督式學習的應用")
    print("Document: docs/1.pdf")
    print("Result:")
    print(table)
    print("Score:", score)
    # main("keyword", "docs/2.pdf")

# %%
