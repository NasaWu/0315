import streamlit as st
import pandas as pd
import numpy as np
import json
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from modelscope import snapshot_download
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import re
import pickle  #to load a saved modelimport base64  #to open .gif files in streamlit app
from functions import retrieve_top_docs
from functions import extract_indices
from functions import get_selected_rows
##################

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

########
@st.cache(suppress_st_warning=True)
def get_fvalue(val):    
	feature_dict = {"No":1,"Yes":2}    
	for key,value in feature_dict.items():        
		if val == key:            
			return value
def get_value(val,my_dict):    
	for key,value in my_dict.items():        
		if val == key:            
			return value
app_mode = st.sidebar.radio('您想询问的问题类型',['第一种问答类型(amp)','第一种问答类型(xxx)']) #two pages

if app_mode == '第一种问答类型(amp)':
    st.title('IPS AMP 操作问答:') 
    # 初始化 session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 显示已有的聊天记录
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # 聊天输入框
    if prompt := st.chat_input("What is up?"):
        top_docs = retrieve_top_docs(prompt)
        indices= extract_indices(top_docs)
        answer = get_selected_rows(df,indices)
        # 将用户的输入添加到聊天记录中
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        # 在这里调用您自己的逻辑来获取回答
        # 假设您的逻辑是一个名为 get_answer 的函数  
        # 显示助手的回答
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.table(answer[['question', 'answer']])