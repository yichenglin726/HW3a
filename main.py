import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            texts.append(table.df.to_string())
        return texts


class text2vector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def transform(self, text):
        return self.vectorizer.transform(text)


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vectors, target_vector):
        return cosine_similarity(vectors, target_vector)


def find_most_relevant_table(tables_text, keyword, vectorizer):
    # 首先使用所有的表格文本來訓練 vectorizer
    vectorizer.fit(tables_text)
    # 將表格文本轉換為向量
    tables_vector = vectorizer.transform(tables_text)
    # 將關鍵字文本轉換為向量
    keyword_vector = vectorizer.transform([keyword])
    # 計算余弦相似度
    similarities = cosine_similarity(tables_vector, keyword_vector)
    most_relevant_index = similarities[:, 0].argmax()
    return tables_text[most_relevant_index]


def main(keyword, pdf_filename):
    pdf_file_path = f"docs/{pdf_filename}"  # Correct relative path
    pdf_parser = pdf2text()
    text_vectorizer = text2vector()
    sim_getter = cosine_sim()

    tables_text = pdf_parser(pdf_file_path)
    most_relevant_table = find_most_relevant_table(tables_text, keyword, text_vectorizer)
    print(most_relevant_table)



if __name__ == "__main__":
    keyword = ""
    pdf_filename = "f"  # 僅文件名稱
    main(keyword, pdf_filename)