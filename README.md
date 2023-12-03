# Stage-A Document Intelligence

## Installation

```bash
conda create -n docint python=3.11
conda activate docint
conda install -c conda-forge ghostscript
pip install -r requirements.txt
```

## Target

Create an artificial intelligence that searching in which table in the given pdf files has the desired information.  

## Usage

1. Exec main.py

```
streamlit run main.py
```

2. A local web page will be displayed

#### Input

1. A fill-in-the-blank box called input a keyword appears below, please enter the searching keywords

#### Output

A most suitable table with desired information in it

#### Example 1

The given pdf file:  
![image](https://github.com/Stage-A/Document-Intelligence/blob/main/images/example1.png)
Search query:

```input following keyword
非監督式學習的應用
```

The Output is as followed:
![image](https://github.com/Stage-A/Document-Intelligence/blob/main/images/example2.png)

#### Example 2
The given pdf file:  
[Document 2](https://docs.google.com/document/d/1HiZrgIyvwY8Fi4eLS0QGUkkycngtD6XJ/edit?usp=sharing&ouid=107784913306655694785&rtpof=true&sd=true)
Search query:

```input following keyword
多細胞生物細胞膜
```

The Output is as followed:
![image](https://github.com/Abclab123/HW3a/blob/51/images/example2out.png)

## How to contribute

- Every one finishes the whole project and pulls the requests , do not edit the main branch
- if your code is acceptable, we will add it into the main branch

## Background Knowledge

[Azure Document Intelligence](https://azure.microsoft.com/en-us/products/ai-services/ai-document-intelligence)

## Test Document

[Document 1](https://docs.google.com/document/d/1Di5oVYhUF6p-zj2y0DEBBeTvhC91KhX8/edit?usp=sharing&ouid=107784913306655694785&rtpof=true&sd=true)
[Document 2](https://docs.google.com/document/d/1HiZrgIyvwY8Fi4eLS0QGUkkycngtD6XJ/edit?usp=sharing&ouid=107784913306655694785&rtpof=true&sd=true)
