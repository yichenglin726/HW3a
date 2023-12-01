import camelot
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")  # 讀取pdf表格
        table_list = []

        for table in tables:

            trans = table.df.replace(
                '\s+|\n', '', regex=True)        # pandas dataframe

            table_list.append(trans)
        count = 0
        for table in table_list:
            if table.shape[0] <= 2 and (count+1) < len(table_list):
                merge1 = table
                merge2 = table_list[count+1]
                result = pd.concat([merge1, merge2], axis=0, ignore_index=True)
                table_list[count] = result
                table_list.pop(count+1)
            count += 1

        return table_list


class text2vector:
    def __init__(self):
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')

    def __call__(self, text):
        return self.model.encode(text)


class cosine_sim:
    def __init__(self):
        self.cos_sim = []
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        for vector in tqdm(vector_from_table, desc="Compute similarity"):
            vector_a = np.array(vector)
            vector_b = np.array([vector_from_keyword])

            dot_product = np.dot(vector_a, vector_b.T)

            norm_a = np.linalg.norm(vector_a)
            norm_b = np.linalg.norm(vector_b)

            cosine_similarity = dot_product / (norm_a * norm_b)

            self.cos_sim.append(cosine_similarity.max())

        return self.cos_sim.index(max(self.cos_sim))


def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)

    vector_trans = text2vector()
    vector_from_table = []
    # dataframe
    for table in tqdm(table_text, desc="Transform tables into vector"):
        texts_vector = []

        for col in table.columns:                                 # each colunmn in the dataframe

            input_texts = table[col].tolist()
            for text in input_texts:
                texts_vector.append(vector_trans(text))
        vector_from_table.append(texts_vector)
    vector_from_keyword = vector_trans(keyword)

    similarity = cosine_sim()
    result = similarity(vector_from_table, vector_from_keyword)
    print("Input Keyword: {k}, Final Result is Table:{i}".format(
        k=keyword, i=result))

    print(table_text[result])
    table_text[result].to_csv("{k}_table_{i}.csv".format(
        k=keyword, i=result), encoding="utf-8", index=False)


if __name__ == "__main__":
    pdf_file = input("請輸入pdf檔案名稱:")
    keyword = input("請輸入欲搜尋之關鍵字:")
    main(keyword, pdf_file)
    # main("非監督式學習的應用", "docs/1.pdf")
    # main("細胞壁", "docs/2.pdf")
