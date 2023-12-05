import camelot
import re    
import pandas as pd
from Levenshtein import distance
import sys
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
        pass

    def __call__(self, text):
        pass


class cosine_sim:
    def __init__(self):
        pass

    def __call__(self, vector_from_table, vector_from_keyword):
        pass
def parse_table_data(input_str):
    data = input_str.split()
    tables = []
    table = []
    row = []
    for item in data:
        if item.isdigit():
            if row:
                table.append(row)
                row = []
            if item == '0' and table:
                tables.append(table)
                table = []
        else:
            row.append(item)
    if row:
        table.append(row)
    if table:
        tables.append(table)
    return tables
def extract_table_info(raw_table_text):
    # Split the raw table text into rows
    raw_rows = raw_table_text.split('\n')
    is_first_row = True
    row_headers = []
    column_headers = []
    row_headers.append([])
    column_headers.append([])
    table_index = 0
    min_word_length = [9999]
    # Iterate over each row
    for raw_row in raw_rows:
        # Split the row into columns
        raw_columns = re.split(r'\s{2,}', raw_row)
        is_header = 0 
        # Iterate over each column
        for raw_column in raw_columns:
            # Remove spaces from the column
            raw_column = raw_column.replace(' ', '')
            if is_header == 0:
                is_header = 1
                # Check if the column is empty
                if raw_column == '':
                    is_header = 0
                    is_first_row = True
                    row_headers.append([])
                    column_headers.append([])
                    min_word_length.append(9999)
                    table_index += 1
                    break
            else:
                # Check if it's the first row
                if is_first_row == True:
                    # Append the column to the row headers
                    row_headers[table_index].append(raw_column)
                    # Update the minimum word length
                    if len(raw_column) < min_word_length[table_index] and len(raw_column) != 0:
                        min_word_length[table_index] = len(raw_column)
                elif is_header == 1:
                    # Append the column to the column headers
                    column_headers[table_index].append(raw_column)
                    # Update the minimum word length
                    if len(raw_column) < min_word_length[table_index] and len(raw_column) != 0:
                        min_word_length[table_index] = len(raw_column)
                    is_header = 2
        # Check if it's the first row
        if is_first_row == True and is_header != 0:
            is_first_row = False
    return row_headers, column_headers, min_word_length

def word_in_query(query, text, word_min_len):
    for i in range(len(query) - (word_min_len-1)):
        q_word = query[i:i+word_min_len]
        for j in range(len(text) - (word_min_len-1)):
            t_word = text[j:j+word_min_len]
            if q_word == t_word:
                return True
    return False

def threshold(word_min_len, query):
    threshold = word_min_len
    threshold -= len(query)
    return abs(threshold)

def find_matching_tables(row_headers, column_headers, min_word_length, search_query):
    # Get the total number of tables
    total_tables = len(row_headers)
    # Remove leading and trailing spaces from the search query
    search_query = search_query.strip()
    matching_table_indices = []
    # Iterate over each table
    for i in range(total_tables):
        min_distance = 9999
        matching_word = 0
        # Iterate over each word in the row headers
        for j in range(len(row_headers[i])):
            word = row_headers[i][j]
            # Calculate the edit distance between the search query and the current word
            d = distance(search_query, word)
            search_query_replace = search_query
            # Update the matching word and the minimum distance if a better match is found
            if d <= min_distance and word_in_query(search_query, word, min_word_length[i]):
                matching_word = word
                min_distance = d
        # Check if the minimum distance is less than the threshold
        if min_distance < threshold(min_word_length[i], search_query):
            min_distance = 9999
            # Remove the matching characters from the search query
            for ch in matching_word:
                search_query_replace = search_query_replace.replace(ch, '', 1)
            # Iterate over each word in the column headers
            for k in range(len(column_headers[i])):
                word = column_headers[i][k]
                # Calculate the edit distance between the remaining search query and the current word
                d = distance(search_query_replace, word)
                # Update the matching word and the minimum distance if a better match is found
                if d <= min_distance and word_in_query(search_query_replace, word, min_word_length[i]):
                    matching_word = word
                    min_distance = d
            # Check if the minimum distance is less than the threshold
            if min_distance < threshold(min_word_length[i], search_query):
                # If a match is found, add the index of the current table to the list
                matching_table_indices.append(i)
    return matching_table_indices

def main(search_query, pdf_file):
    pdf_parser = pdf2text()
    table_text = pdf_parser(pdf_file)
    table = parse_table_data(table_text)
    row_headers, column_headers, min_word_length = extract_table_info(table_text)
    table_index = find_matching_tables(row_headers, column_headers, min_word_length, search_query)
    for i in table_index:
        st.write(pd.DataFrame(table[i-1]))
    st.write('This query has been done.')


if __name__ == "__main__":
    st.write('Student ID: R11944039')
    st.write('Please enter the path of the PDF. Then, enter the word you want to search for.')
    pdf = st.text_input('Path of the PDF', '')
    search_query = st.text_input('Search Word', '')
    
    if len(pdf)>3 and len(search_query)>1:
        st.write('Search "', search_query, '" in ', pdf)
        main(search_query, pdf)
    else:
        st.write(':fire: Please enter your query. :fire:')