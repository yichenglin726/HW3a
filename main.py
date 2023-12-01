import camelot
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import jieba
import argparse
import os


class pdf2text:
    # Extract text from PDF file
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:

            texts.append(table.df)
        
        # replace all '\n' with ''
        for i in range(len(texts)):
            texts[i] = texts[i].replace('\n', '', regex=True)
        
        return texts

class text2vector:
    def __init__(self):
        # Initialize the vectorizer with a custom tokenizer
        self.vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, ngram_range=(1, 1))

    def __call__(self, texts, keyword):
        combined_texts = [' '.join(jieba.cut(' '.join(table.astype(str).values.flatten()))) for table in texts]
        # print(combined_texts)
        # combined_texts.append(' '.join(jieba.cut(keyword)))
        return self.vectorizer.fit_transform(combined_texts), self.vectorizer



class cosine_sim:
    # Calculate cosine similarity
    def __init__(self):
        pass

    def __call__(self, vectors, query_vector):
        # print(vectors.shape)
        similarity_scores = cosine_similarity(query_vector, vectors)
        return similarity_scores[0]


def find_most_relevant_table(keyword, pdf_file):
    """
    Given a keyword and a PDF file, find the table in the PDF that is most relevant to the keyword.

    @param keyword: The keyword to search for.
    @param pdf_file: The PDF file to search in.
    """

    if not os.path.exists(pdf_file):
        print(f"Error: The file '{pdf_file}' does not exist.")
        return
    
    pdf_parser = pdf2text()
    tables = pdf_parser(pdf_file)
    # print(tables)

    vectorizer = text2vector()
    vectors , vv = vectorizer(tables, keyword)
    # print(vectors)

    # The last vector corresponds to the keyword
    keyword_vector = vv.transform([keyword])

    # The rest are the table vectors
    table_vectors = vectors

    # Calculate cosine similarity
    cosine = cosine_sim()
    similarities = cosine(table_vectors, keyword_vector)

    # Find the table with the highest similarity score
    if len(similarities) > 0:
        max_similarity_index = np.argmax(similarities)
        print("Similarity scores: ", similarities)
        if similarities[max_similarity_index] > 0:  # If the highest similarity score is greater than 0
            print("Most relevant table:")
            print(tables[max_similarity_index])
        else:
            print("No relevant table found.")
    else:
        print("No tables found in the PDF.")

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Find the most relevant table in a PDF based on a given keyword.")
    parser.add_argument("keyword", type=str, help="The keyword to search for in the PDF.")
    parser.add_argument("pdf_file", type=str, help="The path to the PDF file.")

    # Parse arguments
    args = parser.parse_args()

    # Run the table finding function
    find_most_relevant_table(args.keyword, args.pdf_file)

if __name__ == "__main__":
    # testing code
    # find_most_relevant_table("非監督式學習的應用", "docs/1.pdf")
    # find_most_relevant_table("動細胞中心體", "docs/2.pdf")

    main()
