import camelot
import numpy as np
import scipy
import streamlit as st
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

class pdf2text:
    """Read pdf and convert to text
    
    Note:
    - Make sure the pdf is in ./docs/ directory.
    """
    def __call__(self, pdf_file):
        tables = camelot.read_pdf(f'docs/{pdf_file.name}', pages="all")
        
        # extract table title
        titles = self.extract_titles(pdf_file)

        texts = [table.df.to_string().replace("\\n", "") for table in tables]
        dfs = [table.df for table in tables]
        return texts, dfs, titles 
    
    def extract_titles(self, pdf_file):
        tables = camelot.read_pdf(f'docs/{pdf_file.name}', flavor="stream", pages="all")
        texts, titles = [], []
        for table in tables:
            texts.append(table.df.to_string().replace("\\n", "").strip().split(" "))

        for text in texts:
            for word in text:
                if "ai_tables" in word:
                    title_idx = text.index(word) + 1
                    title = f"{word} {text[title_idx]}".rstrip()
                    titles.append(title)
        return titles

class text2vector:
    """Create token embeddings for text"""
    def __init__(self):
        # get the model
        self.model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")

    def __call__(self, text):
        # return the token embeddings
        return self.model.encode(text)


class cosine_sim:
    def __call__(self, vector_from_table, vector_from_keyword):
        """Calculate cosine similarity between 2 vectors
        
        Args:
        - vector_from_table: token vector from the input table
        - vector_from_keyword: the keyword to be compared
        """
        # cosine similarity = 1 - cosine distance
        return 1 - scipy.spatial.distance.cosine(vector_from_table, vector_from_keyword)

def search_table(keyword, files):
    """Search for the corresponding table given a keyword
    
    Args:
    - keyword: the keyword to be searched
    - files: the pdf files
    """
    pdf_parser = pdf2text()
    embedding = text2vector()
    cosine_similarity = cosine_sim()

    # embed keyword
    keyword_embedding = embedding(keyword)

    best_similarity, matching_df, matching_title = 0, '', ''
    
    # iterate over input file(s)
    for file in files:
        texts, dfs, titles = pdf_parser(file)
        
        for i, (df, title) in enumerate(zip(dfs, titles)):
            table_embedding = embedding(df.to_string())
            similarity = cosine_similarity(table_embedding, keyword_embedding)

            # update
            if similarity > best_similarity:
                best_similarity = similarity
                matching_df = df
                matching_title = title

    return {'table': matching_df, 'title': matching_title}


class App:
    def run(self):
        """Run streamlit web app"""
        
        # custom styling
        m = st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2BA3FF;
            color:#ffffff;

            /* button alignment and shape */
            display: block;
            margin: 0 auto;
            height: 50px; 
            width: 150px; 
            border-radius: 10px;
        }
        div.stButton > button:hover {
            background-color: #5FADE8;
            color:#000000;
            border-color: #4CAF50;

            /* button alignment and shape */
            display: block;
            margin: 0 auto;
            height: 50px; 
            width: 150px; 
            border-radius: 10px;
        }
        </style>""", unsafe_allow_html=True)

        st.title("Document Intelligence")
        st.write("Big Data Systems HW3a")
        st.divider()

        files = st.file_uploader(
            "Input the pdf file(s) containing tables in which the keyword will be searched in", 
            accept_multiple_files=True
        )

        # handling missing components 
        if "missing_file" in st.session_state and st.session_state.missing_file is not None:
            st.error(st.session_state.missing_file)

        keyword = st.text_input("Type in the keyword to be searched", "學習最佳策略")
        
        # handling missing components 
        if "missing_keyword" in st.session_state and st.session_state.missing_keyword is not None:
            st.error(st.session_state.missing_keyword)
            
        search_button = st.button(":mag_right: Search!", on_click=self.on_click, args=(files, keyword))

        if search_button and \
            st.session_state.missing_file is None and \
            st.session_state.missing_keyword is None:

            with st.spinner("Searching for the table... :face_with_monocle:"):    
                # table_df = search_table(keyword, files)
                matching_info = search_table(keyword, files)
                st.session_state.table = matching_info['table'] 
                st.session_state.title = matching_info['title'] 

                st.subheader(st.session_state.title)
                st.dataframe(st.session_state.table)

    def on_click(self, files, keyword):
        st.session_state["missing_file"] = st.session_state["missing_keyword"] = None
        if not files:
            st.session_state.missing_file = ":warning: Oops! Forget to upload your pdf file?"
        if not keyword:
            st.session_state.missing_keyword = ":warning: Oops! Forget about the keyword?"
        return

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()