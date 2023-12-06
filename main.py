import camelot
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
        self.vectorizer = CountVectorizer()

    def __call__(self, text):
        vector = self.vectorizer.fit_transform([text])
        return vector


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return cosine_similarity(vector_from_table, vector_from_keyword)[0][0]


def find_table_with_keyword(pdf_file, keyword):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)

    vectorizer = text2vector()
    table_vector = vectorizer(table_text)

    keyword_vector = vectorizer(keyword)

    similarity_calculator = cosine_sim()
    similarity = similarity_calculator(table_vector, keyword_vector)

    return table_text, similarity


def main(keywords, pdf_files):
    for pdf_file in pdf_files:
        for keyword in keywords:
            table_text, similarity = find_table_with_keyword(pdf_file, keyword)

            # Adjust the threshold as needed
            if similarity > 0.7:
                print(f"Keyword: {keyword}")
                print(f"PDF File: {pdf_file}")
                print("Table:")
                print(table_text)
                print("Similarity:", similarity)
                print("=" * 50)


if __name__ == "__main__":
    search_keywords = ["desired_info_1", "desired_info_2"]
    pdf_files_to_search = ["docs/1.pdf", "docs/2.pdf"]

    main(search_keywords, pdf_files_to_search)
