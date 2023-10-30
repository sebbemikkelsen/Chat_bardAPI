from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import os
from HC_bot import HC_bot

load_dotenv()


def get_vectorstore(embedding_model, db_name):
    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model)
    db = Chroma(persist_directory=db_name, embedding_function=embeddings)
    return db


def set_custom_prompt(history, context, question):

    #custom_prompt_template = f"""You are a Swedish taxing advisor. Use the context to answer the user's question.
    #If the context can't answer the question, say that you don't know.
    #Chat history: {history}
    ##Context: {context}
    #Question: {question}

    #Only return the helpful answer in swedish, based on the context provided. Otherwise, say "I don't know".
    #Helpful answer:"""

    custom_prompt_template = f"""Du är en Svensk skatterådgivare. Använd den givna lagtexten för att svara på frågorna.
    Om lagen inte kan svara på frågan, säg att du inte vet svaret.
    Historik: {history}
    Lagtext: {context}
    Fråga: {question}

    Ge endast hjälpsamma svar på svenska, baserat på lagtexten. Annars, säg "Jag vet inte".
    Hjälpsamt svar:"""

    return custom_prompt_template


def get_matching_docs(question):
    db_directories = [f"test_chroma_db_batch_{x}" for x in range(11)]
    embeddings = HuggingFaceEmbeddings(model_name="KBLab/sentence-bert-swedish-cased")
    matching_docs = []

    # Perform similarity search on each database
    for db_directory in db_directories:
        db = Chroma(persist_directory=db_directory, embedding_function=embeddings)
        #matching_docs += db.similarity_search(question)
        #matching_docs.append(db.similarity_search(question)[0])
        #print(db.similarity_search(question))
        #print('===============================================================')
        #print('===============================================================')

        doc = db.similarity_search(question)
        if len(doc) > 0:
            #matching_docs.append(db.similarity_search(question)[0])
            matching_docs.append(db.similarity_search(question)[0])

    return matching_docs


def create_chatbot():
    email = os.getenv("EMAIL_HF")
    pw = os.getenv("PASS_HF")
    bot = HC_bot(email, pw)
    bot.new_chat()

    return bot


def ask_bot(question, history, source=False):
    matching_docs = get_matching_docs(question)
    context = matching_docs[0].page_content + matching_docs[1].page_content
    #context = ""
    #for doc in matching_docs: 
    #    context += doc.page_content
    #print(matching_docs)
    #print(context)

    prompt = set_custom_prompt(history, context, question)

    bot = create_chatbot()
    ans = bot.one_chat(prompt)

    if source:
        return ans, matching_docs[0].metadata['source'], matching_docs[1].metadata['source']
    else:
        return ans
    



def chat():
    history = ""
    query = input(">>> ")

    while query != "quit":
        ans = ask_bot(query, history)
        add_to_hist = query + ans
        history = history + add_to_hist

        print("=====================================================")
        print(ans)
        print("=====================================================")

        query = input(">>> ")




def main():

    chat()


if __name__=='__main__':
    main()

