# LocalRAG
Local and private RAG system, inspired by privateGPT.

Uses langchain, llama-cpp, txtai.

# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip3 install -r requirements.txt
```

Edit the variables appropriately in the `.env` file.
```
PERSIST_DIRECTORY = Where you want your vector store and sqlite3 database to be
MODEL_TYPE = Model type used, only supports LlamaCpp
MODEL_PATH = path to your model
EMBEDDINGS_MODEL_NAME = embeddings model name, not used
MODEL_N_CTX = context window size for your model
TARGET_SOURCE_CHUNKS = number of chunks returned from txtai's hybrid search
IS_GPU_ENABLED=whether or not a GPU is available
MAX_TOKENS_GENERATED = max number of tokens the model will  generate
MODEL_TEMPERATURE = model temperature, a value between 0 and 1. higher temps mean more randomness
CHUNK_SIZE = chunk size when ingesting documents
CHUNK_OVERLAP = chunk overlap when ingesting documents
SOURCE_DIRECTORY = name of the folder where source documents will be stored
```

## Instructions for ingesting your own dataset

Put any and all your files into the `source_documents` directory

The supported extensions are:

   - `.csv`: CSV,
   - `.docx`: Word Document,
   - `.doc`: Word Document,
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.html`: HTML File,
   - `.md`: Markdown,
   - `.msg`: Outlook Message,
   - `.odt`: Open Document Text,
   - `.pdf`: Portable Document Format (PDF),
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.txt`: Text file (UTF-8),

Run the following command to ingest all the data.

```shell
python ingest.py
```

Output should look like this:

```shell
Creating new vectorstore
Loading documents from source_documents
Loading new documents: 100%|██████████████████████| 1/1 [00:01<00:00,  1.73s/it]
Loaded 1 new documents from source_documents
Split into 90 chunks of text (max. 500 tokens each)
Creating embeddings. May take some minutes...
Using embedded DuckDB with persistence: data will be stored in: db
Ingestion complete! You can now run LocalRag.py to query your documents
```

It will create a `db` folder containing the local vectorstore. Will take 20-30 seconds per document, depending on the size of the document.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `db` folder.

## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python LocalRag.py
```
Navigate to the URL printed to stdout by the script. Use the UI to ask questions. Once done, the UI will display your answer and the terminal will display the chunks of source document used as context.


# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.
