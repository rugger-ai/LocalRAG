#!/usr/bin/env python3
from txtai import Embeddings
import gradio as gr
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import argparse
from torch import cuda as torch_cuda
from langchain import PromptTemplate
from langchain.chains import LLMChain
import sqlite3
import pprint
import time


load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path_RAG = os.environ.get('MODEL_PATH_RAG')
model_path_chat = os.environ.get('MODEL_PATH_CHAT')
model_path_code = os.environ.get('MODEL_PATH_CODE')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_temperature_RAG = os.environ.get('MODEL_TEMPERATURE_RAG')
model_temperature_chat = os.environ.get('MODEL_TEMPERATURE_CHAT')
model_temperature_code = os.environ.get('MODEL_TEMPERATURE_CODE')
max_tokens_generated = os.environ.get('MAX_TOKENS_GENERATED')
is_gpu_enabled = (os.environ.get('IS_GPU_ENABLED', 'False').lower() == 'true')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

def get_gpu_memory() -> int:
    """
    Returns the amount of free memory in MB for each GPU.
    """
    return int(torch_cuda.mem_get_info()[0]/(1024**2))

def calculate_layer_count() -> int | None:
    """
    Calculates the number of layers that can be used on the GPU.
    """
    if not is_gpu_enabled:
        return None
    LAYER_SIZE_MB = 120.6 # This is the size of a single layer on VRAM, and is an approximation.
    # The current set value is for 7B models. For other models, this value should be changed.
    LAYERS_TO_REDUCE = 6 # About 700 MB is needed for the LLM to run, so we reduce the layer count by 6 to be safe.
    if (get_gpu_memory()//LAYER_SIZE_MB) - LAYERS_TO_REDUCE > 32:
        return 32
    else:
        return (get_gpu_memory()//LAYER_SIZE_MB-LAYERS_TO_REDUCE)

mode = input("Enter a mode ('rag', 'chat', 'code'): ")

if mode == "rag":
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """

    llamallm = LlamaCpp(model_path=model_path_RAG, streaming=True, max_tokens = max_tokens_generated, temperature = model_temperature_RAG, n_ctx = model_n_ctx, verbose=False, n_gpu_layers=calculate_layer_count())
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    print("Loading Embeddings model....")
    embeddings = Embeddings(hybrid=True, path="sentence-transformers/nli-mpnet-base-v2")

    #load the embeddings from saved embeddings
    print("Loading Embeddings...")
    embeddings.load("db")

if mode == "chat":
    prompt_template = """You are a helpful virtual assistant. 

    Question: {question}
    Answer:
    """
    llamallm = LlamaCpp(model_path=model_path_chat,  streaming=True, max_tokens = max_tokens_generated, temperature = model_temperature_chat, n_ctx = model_n_ctx, verbose=False, n_gpu_layers=calculate_layer_count())
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question"]
    )

if mode == "code":
    prompt_template = """You are a helpful virtual assistant for programmers. 

    Question: {question}
    Answer:
    """
    llamallm = LlamaCpp(model_path=model_path_code,  streaming=True, max_tokens = max_tokens_generated, temperature = model_temperature_code, n_ctx = model_n_ctx, verbose=False, n_gpu_layers=calculate_layer_count())
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["question"]
    )


while True:
    question = input("Enter a query: ")

    start_time = time.time()
    
    if question != "" and len(question) > 1:
        # Extract uid of first result
        # search result format: (uid, score)
        # return the top N results
        if mode == 'rag':
            connection = sqlite3.connect("db/embeddings.db")
            cursor = connection.cursor()
            uid = embeddings.search(question, target_source_chunks)

            embeddingSearchResults = ""
            
            for uid, score in uid:
                #print("  %s" % data[uid])
                cursor.execute(
                'SELECT * FROM chunks LIMIT 1 offset (?)', (uid,))
                results = cursor.fetchall()
                embeddingSearchResults += "\n\n" + str(results)

            print("Embeddings Search Results:")
            #embeddingSearchResults = embeddingSearchResults.replace(r'\n', '\n')
            print(embeddingSearchResults)


        llmChain = LLMChain(llm=llamallm, prompt=PROMPT)
        if mode == "rag":
            result = llmChain.run(question=question, context=embeddingSearchResults)
        else: 
            result = llmChain.run(question=question)
        run_time = time.time() - start_time
        print("Time to execute: " + str(round(run_time)) + " seconds")
        print(result)
    

