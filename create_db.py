from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.document_loaders import TextLoader


directory = './subpage_texts'

def load_docs(directory):
  """load all the documents from the specified directory"""
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


def load_txt_files(directory):

    loader = TextLoader("./index.md")
    documents = loader.load()


def split_docs(documents,chunk_size=500,chunk_overlap=10):
  """Split the documents in smaller text chunks, with some overlap to not loose contex"""
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs



def store_vectorstore(docs, db_name):
    """Store the text vectors in the database in batches."""
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

    batch_size = 166  # Set the batch size to a value below the limit

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i + batch_size]
        db = Chroma.from_documents(batch_docs, embeddings)
        persist_directory = f"{db_name}_batch_{i // batch_size}"
        vectordb = Chroma.from_documents(
            documents=batch_docs, embedding=embeddings, persist_directory=persist_directory
        )
        vectordb.persist()


"""
#def store_vectorstore(docs, db_name):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    #embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

    db = Chroma.from_documents(docs, embeddings)
    persist_directory = db_name
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()
"""



def main():
    db_name = "test_chroma_db"
    documents = load_docs(directory)
    #documents = load_txt_files(directory)
    docs = split_docs(documents)
    print(docs[1])
    store_vectorstore(docs, db_name)



if __name__ == '__main__':
    main()





