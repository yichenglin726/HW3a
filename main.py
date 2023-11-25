import camelot
from transformers import BertModel, BertTokenizer
import numpy as np

# Load the BERT model and tokenizer
model_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

class PDF2Text:
    def __call__(self, pdf_file):
        try:
            tables = camelot.read_pdf(pdf_file, pages="all")
        except Exception as e:
            print(f"Error reading PDF file: {e}")
            return ""
        return "\n".join([table.df.to_string().replace("\\n", "") for table in tables])

def text2vector(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding='max_length')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def cosine_sim(vec1, vec2):
    vec1, vec2 = vec1.flatten(), vec2.flatten()
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def process_table_text(table_text):
    return table_text.replace("\n ", "\n\n")

def split_strings(table):
    return table.replace('\n', ' ').split()

def find_most_relevant_tables(keyword, pdf_file, top_n=2):
    pdf_parser = PDF2Text()
    table_text = pdf_parser(pdf_file)

    if not table_text:
        return "No tables found or error in reading PDF."

    table_text = process_table_text(table_text)
    tables = table_text.split('\n\n')

    keyword_vector = text2vector(keyword)
    table_similarities = []

    for table in tables:
        similarities = [cosine_sim(keyword_vector, text2vector(string)) for string in split_strings(table)]
        if similarities:
            top_similarities = sorted(similarities, reverse=True)[:top_n]
            mean_similarity = sum(top_similarities) / len(top_similarities)
            table_similarities.append((mean_similarity, table))

    # Sort the tables by average similarity and get the top N
    best_tables = [table for _, table in sorted(table_similarities, key=lambda x: x[0], reverse=True)[:top_n]]

    return best_tables if best_tables else ["No table closely matches the keyword."]

if __name__ == "__main__":
    keyword = "中空的細胞器"
    pdf_file = "docs/2.pdf"
    matched_tables = find_most_relevant_tables(keyword, pdf_file)
    for table in matched_tables:
        print(table)

