from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
import os 

load_dotenv()







from langchain.embeddings import HuggingFaceEmbeddings






"""



directory = './docs'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents


documents = load_docs(directory)


def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def store_vectorstore(docs):
    #embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    db = Chroma.from_documents(docs, embeddings)
    persist_directory = "chroma_db_swe_1"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()
    return db

docs = split_docs(documents)
#db = store_vectorstore(docs)
"""







# Define the list of database directories
db_directories = ["./chroma_db_swe1_batch_0", "./chroma_db_swe1_batch_1", "./chroma_db_swe1_batch_2", "./chroma_db_swe1_batch_3", "./chroma_db_swe1_batch_4", "./chroma_db_swe1_batch_5", "./chroma_db_swe1_batch_6", "./chroma_db_swe1_batch_7"]
embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")

# Define your query
query = "När anses bolaget bildat?"


# Initialize the list to store matching documents
matching_docs = []


# Perform similarity search on each database
for db_directory in db_directories:
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    matching_docs += db.similarity_search(query)


"""
# Set batch size
batch_size = 10  # Adjust the batch size as needed

# Perform similarity search on each database in batches
for db_directory in db_directories:
    db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
    
    # Iterate through the documents in batches
    for i in range(0, len(db.documents), batch_size):
        batch_docs = db.documents[i:i + batch_size]
        batch_matching_docs = db.similarity_search(query, documents=batch_docs)
        matching_docs += batch_matching_docs
"""




#embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
#embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")
#db = Chroma(persist_directory="./chroma_db3", embedding_function=embeddings)

#query = "How many books is there in the library?"
#query = "How many nations are there?"
#query = input("Question: ")
#matching_docs = db.similarity_search_with_score(query, k=4)
#matching_docs = db.similarity_search(query)



























HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

#repo_id = "google/flan-t5-xxl"
#repo_id = "timpal0l/mdeberta-v3-base-squad2"
#repo_id = "google/mt5-large"
#repo_id = "google/mt5-base"


#llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5}) #, "min_tokens": 200




"""
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

#answer = chain.run(input_documents=matching_docs, question=query)
#answer = chain({"input_documents": matching_docs, "question": query}, return_only_outputs=True)

#print(answer)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
print("Answer: ", retrieval_chain.run(query))
print("Source of information: ", matching_docs[0].metadata)
"""






import requests
au = "Bearer " + HUGGINGFACEHUB_API_TOKEN

API_URL = "https://api-inference.huggingface.co/models/timpal0l/mdeberta-v3-base-squad2"
headers = {"Authorization": au}

def query1(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query1({
	"inputs": {
		"question": query,
		"context": matching_docs[0].page_content
	},
})

print(output)







"""
#retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
#qa = RetrievalQA.from_chain_type(
#    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


chat_history = []
while True:
   query = input("Question")
   if query == "quit":
      break
   
    # Initialize the list to store matching documents
   matching_docs = []

  # Perform similarity search on each database
   for db_directory in db_directories:
      db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
      matching_docs += db.similarity_search(query)

   retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
   qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

   result = qa({"query": query, "chat_history": chat_history})
   print("-----------------------------------------")
   print("-----------------------------------------")
   print("ANS: ", result["result"])
   print("-----------------------------------------")
   print("-----------------------------------------")
   hist = (query, result)
   chat_history.append(hist)
"""

























"""
query = "how many books are there in the library?"
chat_history = []
result = qa({"query": query, "chat_history": chat_history})

print("ANS1: ", result)
print("-------------------------")
print("-------------------------")
print("-------------------------")

chat_history = [(query, result["result"])]
query = "What else does the library contain?"
result = qa({"query": query, "chat_history": chat_history})

print("Chat hist: ", chat_history)
print("-------------------------")
print("-------------------------")
print("-------------------------")
print(result)
"""






"""
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(llm, retriever)
chat_history = []
query = "how many books are there in the library?"
result = qa({"question": query, "chat_history": chat_history})

print("ANS1: ", result)

chat_history = [(query, result["answer"])]
query = "What else does the library contain?"
result = qa({"question": query, "chat_history": chat_history})

print("Chat hist: ", chat_history)
print(result['answer'])

"""



# Read document


#create LLM


# send doc to LLM


# ask question

