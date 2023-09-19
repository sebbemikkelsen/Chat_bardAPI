from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import os 


directory = './docs'

def load_docs(directory):
  """load all the documents from the specified directory"""
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


documents = load_docs(directory)


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  """Split the documents in smaller text chunks, with some overlap to not loose contex"""
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


docs = split_docs(documents)


def store_vectorstore(docs):
    """Store the text vectors in database"""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(docs, embeddings)
    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()
    return db

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
#db = store_vectorstore(docs)


#query = "How many books is there in the library?"
#query = "How many nations are there?"
query = input("Question: ")
#matching_docs = db.similarity_search_with_score(query, k=4)
matching_docs = db.similarity_search(query)






load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "google/flan-t5-xxl"
#repo_id = "google/mt5-base"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

#answer = chain.run(input_documents=matching_docs, question=query)
#answer = chain({"input_documents": matching_docs, "question": query}, return_only_outputs=True)

#print(answer)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
print("Answer: ", retrieval_chain.run(query))
print("Source of information: ", matching_docs[0].metadata)









# Read document


#create LLM


# send doc to LLM


# ask question

