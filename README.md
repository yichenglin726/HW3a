# Stage-A Document Intelligence: PDF Table Search Engine
資訊工程所碩一, R12922050, 郭子麟, Branch `24` on the hw3a sheet

## Basic Introduction
This application is designed to extract tables from a PDF, accept query keywords, and search for the most correlated table to return on the screen.

## Changes Made

The main changes in this project include:

- Added a PDF to text parser using Camelot to extract tables from PDFs.
- Implemented a text to vector converter using the `SentenceTransformer` with `distiluse-base-multilingual-cased-v2` model being used.
- Calculated cosine similarity between vectors to find the most correlated table.
- Created a Streamlit application to interactively select a PDF, input a keyword, and display the most correlated table.
- Added cute and joyful search messages that are displayed randomly while the search is in progress.

## Flowchart Diagram
![flow drawio](https://github.com/Abclab123/HW3a/assets/81730878/a091c124-981a-4c1c-843d-fe32560b9028)


## How to Run the Program

1. Install the required Python packages:

  ```bash
conda create -n docint python=3.11
conda activate docint
conda install -c conda-forge ghostscript
pip install -r requirements.txt
  ```

2. Run the streamlit UI
```bash
streamlit run main.py
```

3. Open your web browser and go to http://localhost:8501 to view the application.
![image](https://github.com/Abclab123/HW3a/assets/81730878/641097a4-37a8-4cee-9e8c-21ad473b946d)
