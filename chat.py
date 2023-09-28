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


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)



query = "How many books is there in the library?"
#query = "How many nations are there?"
#query = input("Question: ")
#matching_docs = db.similarity_search_with_score(query, k=4)
matching_docs = db.similarity_search(query)



HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

repo_id = "google/flan-t5-xxl"
#repo_id = "google/mt5-base"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "min_tokens": 200})




"""
chain = load_qa_chain(llm, chain_type="stuff",verbose=True)

#answer = chain.run(input_documents=matching_docs, question=query)
#answer = chain({"input_documents": matching_docs, "question": query}, return_only_outputs=True)

#print(answer)

retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=db.as_retriever())
print("Answer: ", retrieval_chain.run(query))
print("Source of information: ", matching_docs[0].metadata)
"""




retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)


chat_history = []
while True:
   query = input("Question")
   if query == "quit":
      break
   result = qa({"query": query, "chat_history": chat_history})
   print("-----------------------------------------")
   print("-----------------------------------------")
   print("ANS: ", result["result"])
   print("-----------------------------------------")
   print("-----------------------------------------")
   hist = (query, result)
   chat_history.append(hist)

   

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

