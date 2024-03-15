import pandas as pd
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from modelscope import snapshot_download
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import gradio as gr
import re

####################################################################################
# D:/Programe/AIGC/0312/
loader = CSVLoader(file_path='D:/Programe/AIGC/0312/test0312question.csv', encoding='gbk')
documents = loader.load()
df = pd.read_csv('D:/Programe/AIGC/0312/test0312.csv', encoding='gbk')
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
docs = text_splitter.split_documents(documents)
model_dir = snapshot_download("AI-ModelScope/bge-large-zh-v1.5", revision='master')
embedding_path=model_dir
embeddings = HuggingFaceBgeEmbeddings(model_name = embedding_path)

vectorstore = FAISS.from_documents(
    docs,
    embedding= embeddings
)
retriever = vectorstore.as_retriever()


def retrieve_top_docs(query_vector, top_k=2):
    return retriever.invoke(query_vector, top_k=top_k)

def extract_indices(top_docs):
    indexlist = []
    for doc in top_docs:
        content_lines = doc.page_content.split('\n')
        for line in content_lines:
            if line.startswith('index:'):
                index = int(line.split(':')[1].strip())
                indexlist.append(index)
    return indexlist

def get_selected_rows(df, indexlist):
    return df.iloc[indexlist].reset_index(drop=True)

def format_data(selected_rows):
    formatted_data = []
    for index, row in selected_rows.iterrows():
        formatted_data.append({
            "question": row["question"].replace("\n", "<br>"),
            "answer": row["answer"].replace("\n", "<br>")
        })
    return formatted_data

def format_json_question(formatted_data):
    formatted_question = []
    for item in formatted_data:
        formatted_q = {
            "Q": f'<span style="color: red;">{item["question"]}</span>'
            # "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_question.append(formatted_q)
    return formatted_question

def format_json_answer(formatted_data):
    formatted_answer = []
    for item in formatted_data:
        formatted_a = {
            # "Q": f'<span style="color: red;">{item["question"]}</span>'
            "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_answer.append(formatted_a)
    return formatted_answer

def format_json_result(formatted_data):
    formatted_result = []
    for item in formatted_data:
        formatted_item = {
            "Q": f'<span style="color: red;">{item["question"]}</span>',
            "A": f'<span style="color: green;">{item["answer"]}</span>'
        }
        formatted_result.append(formatted_item)
    return formatted_result

def qa_system(query_vector, df):
    top_docs = retrieve_top_docs(query_vector)
    indexlist = extract_indices(top_docs)
    selected_rows = get_selected_rows(df, indexlist)
    formatted_data = format_data(selected_rows)
    formatted_question = format_json_question(formatted_data)
    formatted_answer = format_json_answer(formatted_data)
    formatted_result = format_json_result(formatted_data)
    return formatted_result