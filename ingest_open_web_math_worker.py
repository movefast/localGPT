import os
os.environ["CURL_CA_BUNDLE"]=""
os.environ["REQUESTS_CA_BUNDLE"]="/etc/ssl/certs/ca-certificates.crt"
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import click
import torch
from langchain.docstore.document import Document
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from utils import get_embeddings
import asyncio
import json
import sys
from langchain_community.vectorstores import Qdrant


from constants import (
    CHROMA_SETTINGS,
    DOCUMENT_MAP,
    EMBEDDING_MODEL_NAME,
    INGEST_THREADS,
    PERSIST_DIRECTORY,
    SOURCE_DIRECTORY,
)

QDRANT_URL = "10.193.192.113"


def file_log(logentry):
    file1 = open("file_ingest.log", "a")
    file1.write(logentry + "\n")
    file1.close()
    print(logentry + "\n")


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    try:
        file_extension = os.path.splitext(file_path)[1]
        loader_class = DOCUMENT_MAP.get(file_extension)
        if loader_class:
            file_log(file_path + " loaded.")
            loader = loader_class(file_path)
        else:
            file_log(file_path + " document type is undefined.")
            raise ValueError("Document type is undefined")
        return loader.load()[0]
    except Exception as ex:
        file_log("%s loading error: \n%s" % (file_path, ex))
        return None


def load_document_batch(filepaths):
    logging.info("Loading document batch")
    # create a thread pool
    with ThreadPoolExecutor(len(filepaths)) as exe:
        # load files
        futures = [exe.submit(load_single_document, name) for name in filepaths]
        # collect data
        if futures is None:
            file_log(name + " failed to submit")
            return None
        else:
            data_list = [future.result() for future in futures]
            # return data and file paths
            return (data_list, filepaths)


def load_documents(source_dir: str) -> list[Document]:
    # Loads all documents from the source documents directory, including nested folders
    paths = []
    for root, _, files in os.walk(source_dir):
        for file_name in files:
            print("Importing: " + file_name)
            file_extension = os.path.splitext(file_name)[1]
            source_file_path = os.path.join(root, file_name)
            if file_extension in DOCUMENT_MAP.keys():
                paths.append(source_file_path)

    # Have at least one worker and at most INGEST_THREADS workers
    n_workers = min(INGEST_THREADS, max(len(paths), 1))
    chunksize = round(len(paths) / n_workers)
    docs = []
    with ProcessPoolExecutor(n_workers) as executor:
        futures = []
        # split the load operations into chunks
        for i in range(0, len(paths), chunksize):
            # select a chunk of filenames
            filepaths = paths[i : (i + chunksize)]
            # submit the task
            try:
                future = executor.submit(load_document_batch, filepaths)
            except Exception as ex:
                file_log("executor task failed: %s" % (ex))
                future = None
            if future is not None:
                futures.append(future)
        # process all results
        for future in as_completed(futures):
            # open the file and load the data
            try:
                contents, _ = future.result()
                docs.extend(contents)
            except Exception as ex:
                file_log("Exception: %s" % (ex))

    return docs


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, python_docs = [], []
    for doc in documents:
        if doc is not None:
            file_extension = os.path.splitext(doc.metadata["source"])[1]
            if file_extension == ".py":
                python_docs.append(doc)
            else:
                text_docs.append(doc)
    return text_docs, python_docs


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    # logging.info(f"Loading documents from {SOURCE_DIRECTORY}")
    # documents = load_documents(SOURCE_DIRECTORY)
    # text_documents, python_documents = split_documents(documents)
    from datasets import load_dataset
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=256, chunk_overlap=32, disallowed_special=())
    # python_splitter = RecursiveCharacterTextSplitter.from_language(
    #     language=Language.PYTHON, chunk_size=880, chunk_overlap=200
    # )
    # texts = text_splitter.split_documents(text_documents)
    # texts.extend(python_splitter.split_documents(python_documents))
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME}")
    from pymongo import MongoClient
    client = MongoClient(host="10.193.192.113")
    db = client.jobs
    col = db['emb_open_world_math_0223']
    
    while jobs := list(col.aggregate([
        { "$match": { 'status': 0 } },
        { "$sample": { 'size': 1 } }
    ])):
        job_meta = jobs[0]
        batch_idx = job_meta['_id']
        try:
            col.update_one({'_id': batch_idx}, {"$set": {"status": 1}})
            open_web_math = load_dataset("parquet", data_dir="/home/derek/Datasets/open-web-math/data", split="train")

            batch = open_web_math.shard(num_shards=6114, index=batch_idx)
            logging.info(f"=== Batch {batch_idx} begins ===")
            docs = text_splitter.create_documents(texts=batch['text'],metadatas=[{
                "url": x['url'], 
                "math_score": json.loads(x['metadata'])['extraction_info']['math_score'],
                "perplexity": json.loads(x['metadata'])['extraction_info']['perplexity']
            } for x in batch])
            texts = text_splitter.transform_documents(docs)
            logging.info(f"Loaded {len(batch['url'])} documents from open_web_math dataset")
            logging.info(f"Split into {len(texts)} chunks of text")

            """
            (1) Chooses an appropriate langchain library based on the enbedding model name.  Matching code is contained within fun_localGPT.py.
            
            (2) Provides additional arguments for instructor and BGE models to improve results, pursuant to the instructions contained on
            their respective huggingface repository, project page or github repository.
            """

            Qdrant.from_documents(
                texts,
                embeddings,
                url=QDRANT_URL,
                prefer_grpc=True,
                collection_name="open_web_math_instructor_large",
            )
            col.update_one({'_id': batch_idx}, {"$set": {"status": 2}})
        except KeyboardInterrupt:
            col.update_one({'_id': batch_idx}, {"$set": {"status": 0}})
            sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
