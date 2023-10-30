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

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_temperature = os.environ.get('MODEL_TEMPERATURE')
max_tokens_generated = os.environ.get('MAX_TOKENS_GENERATED')
is_gpu_enabled = (os.environ.get('IS_GPU_ENABLED', 'False').lower() == 'true')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# Open and split the source document
data = open("titanic_json.txt").read().split("}")

print("Creating Embeddings....")

# Create embeddings
embeddings = Embeddings(hybrid=True, path="sentence-transformers/nli-mpnet-base-v2")

print("Indexing Embeddings...")

# Create an index for the embeddings
embeddings.index(data)

# Instantiate the LLM
llamallm = LlamaCpp(model_path=model_path, max_tokens = max_tokens_generated, temperature = model_temperature, n_ctx = model_n_ctx, verbose=False, n_gpu_layers=15)

# Create the prompt template for the LLM
prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:
    """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def talk_to_LLM(question, history):

    # Extract uid of first result
    # search result format: (uid, score)
    # return the top N results
    uid = embeddings.search(question, 5)

    embeddingSearchResults = ""
    
    for uid, score in uid:
        #print("  %s" % data[uid])
        embeddingSearchResults += data[uid]

    print("Embeddings Search Results:")
    print(embeddingSearchResults)
    
    if question != "":
        llmChain = LLMChain(llm=llamallm, prompt=PROMPT)
        return llmChain.run(question=question, context=embeddingSearchResults)
        #print("\n\nQuestion: " + question + "\n\nAnswer: " + llmChain.run(question=question, context=embeddingSearchResults)  + "\n\n")


gr.ChatInterface(talk_to_LLM).launch()