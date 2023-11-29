# Import necessary libraries
import camelot
import streamlit as st
import os
import pandas as pd
import pdfplumber
import re
import jieba
from gensim import corpora, models, similarities

# Class for extracting text and tables from PDF files
class pdf2text:

    def __init__(self):
        pass

    def __call__(self, pdf_file):
        # Extract tables using camelot
        pdf_tables = camelot.read_pdf(pdf_file, pages="all")
        tables = []
        titles = []
        texts = []
        for table in pdf_tables:

            # Clean and format the table data
            table_df = table.df
            nan_value = float("NaN")
            table_df.replace('', nan_value, inplace=True)
            table_df = table_df.dropna(axis=0, how='all', inplace=False)
            table_df = table_df.reset_index(drop=True)

            # Remove newlines and spaces from table data
            for series in table_df:
                table_df[series] = table_df[series].str.replace("\n", "")
                table_df[series] = table_df[series].str.replace(" ", "")

            # Convert table to string for text processing
            table_str = table_df.to_string()
            table_str = table_str.replace(" ","、")
            table_str = table_str.replace("\n","、")

            # Append tables and texts for further processing
            if table_df[0][0] != "特點":
                tables[-1] = pd.concat([tables[-1], table_df])
                tables[-1] = tables[-1].reset_index(drop=True)
                texts[-1] = texts[-1] + table_str
            else:
                tables.append(table_df)
                table_df = table_df.reset_index(drop=True)
                texts.append(table_str)

        # Extract text using pdfplumber
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                pattern = re.compile(r'^(ai_tables_\S+)', re.MULTILINE)
                matches = pattern.findall(page_text)

                # Extract titles from the text
                if matches:
                    titles.extend(matches)

        # Combine titles with extracted texts
        for i in range(len(texts)):
            texts[i] = titles[i] + texts[i]
        

        for df in range(len(tables)):
            names = tables[df].loc[0, :].values.flatten().tolist()
            tables[df].columns = names
            tables[df] = tables[df].drop([0]).reset_index(drop=True)

        return tables, titles, texts

# Class for converting text to a vector of words
class text2vector:

    def __init__(self):
        pass

    def __call__(self, text):
        # Use jieba for Chinese word segmentation
        words = jieba.cut(text, cut_all=False)

        # Filter out stop words
        stop_words = {}.fromkeys(
            [line.rstrip() for line in open('stopwords.txt', encoding='utf8')])
        filtered_words = [word for word in words if word not in stop_words]
        # filtered_words = [word for word in words ]
        return filtered_words

# Class for calculating cosine similarity between vectors
class cosine_sim:

    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        # Create a dictionary from the vectors
        dictionary = corpora.Dictionary(vector_from_table)
        # Convert documents to vector space
        corpus = [dictionary.doc2bow(doc) for doc in vector_from_table]
        doc_keyword = dictionary.doc2bow(vector_from_keyword)
        # Create a TF-IDF model and index
        tfidf = models.TfidfModel(corpus)
        index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=len(dictionary.keys()))  
        # Calculate similarity
        sim = index[tfidf[doc_keyword]]
        # Get the most similar document
        
        (key_index, value) = sorted(enumerate(sim), key=lambda item: -item[1])[0]
        print(sorted(enumerate(sim), key=lambda item: -item[1]))
        # print(vector_from_table, vector_from_keyword)
        return key_index, value

# Main function to orchestrate PDF text extraction and similarity calculation
def main(sentence, pdf_files):
    tables = []
    titles = []
    texts = []
    table_text = pdf2text()
    t2v = text2vector()
    get_index = cosine_sim()

    # Process each PDF file
    for pdf_file in pdf_files:
        table, title, text = table_text("docs/" + pdf_file)
        tables = tables + table
        titles = titles + title
        texts = texts + text
    # Convert texts to vectors
    text_vec = [t2v(text) for text in texts]
    key_vec = t2v(sentence)

    # Find the most similar text
    index, sim = get_index(text_vec, key_vec)

    title = titles[index]
    table = tables[index]

    return title, table

# Streamlit app setup
if __name__ == "__main__":

    # Setup for selecting PDF files
    dir_path = os.path.dirname(os.path.realpath(__file__)) + "/docs"
    files = os.listdir(dir_path)
    st.title('AI PDF Table Search Tool')

    # Dropdown for file selection
    selected_files = st.multiselect("Select files:",
                             files,
                             default=None,
                             key=None)

    # Input for search keywords
    input_keywords = st.text_input("Enter keywords:")

    submit = st.button("Search")

    # Handling search button click
    if submit:
        if not input_keywords or not selected_files:
            # Display error if input is invalid
            st.error("Please enter keywords and select at least one PDF file.")
        else:
            # Perform search and display results
            with st.spinner('Doing some magic....'):
                title, table = main(input_keywords, selected_files)
                st.subheader(title)
                st.table(table)
