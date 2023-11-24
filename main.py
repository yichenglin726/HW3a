import camelot
import math
import re
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial import distance

class pdf2text:
    def __init__(self):
        pass

    def __call__(self, pdf_file):
        tables = camelot.read_pdf(pdf_file, pages="all")
        texts = []
        for table in tables:
            text = table.df.to_string()
            text = text.replace("\\n", "")
            texts.append(text)
        return texts


class text2vector:
    def __init__(self):
        pass

    def __call__(self, text):
        # sentences to list
        allsentences = [text]
    
        # text to vector
        vectorizer = CountVectorizer()
        all_sentences_to_vector = vectorizer.fit_transform(allsentences)
        return all_sentences_to_vector.toarray().tolist()


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        return 1-distance.cosine(vector_from_table, vector_from_keyword)

def cosine_distance_countvectorizer_method(s1, s2):
    
    # sentences to list
    allsentences = [s1 , s2]
 
    # text to vector
    vectorizer = CountVectorizer()
    all_sentences_to_vector = vectorizer.fit_transform(allsentences)
    text_to_vector_v1 = all_sentences_to_vector.toarray()[0].tolist()
    text_to_vector_v2 = all_sentences_to_vector.toarray()[1].tolist()
    
    # distance of similarity
    cosine = distance.cosine(text_to_vector_v1, text_to_vector_v2)
    #print('Similarity of two sentences are equal to ',round((1-cosine)*100,2),'%')
    return 1-cosine

def main(keyword, pdf_file):
    pdf_parser = pdf2text()
    tables_text = pdf_parser(pdf_file)
    cos_sim_val = []

    for table_text in tables_text:
        cos_sim_val.append(cosine_distance_countvectorizer_method(keyword, table_text))
    
    key = cos_sim_val.index(max(cos_sim_val))

    #print(cos_sim_val)
    print(tables_text[key])
    return tables_text[key]


if __name__ == "__main__":
    main("非監督式學習的應用", "docs/1.pdf")
    main("動物細胞", "docs/2.pdf")
