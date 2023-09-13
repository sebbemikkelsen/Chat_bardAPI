from dotenv import load_dotenv
from langchain import HuggingFaceHub, LLMChain
from langchain.prompts import PromptTemplate
import os 

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

prompt = PromptTemplate(
    input_variables=["question"], 
    template = "Question: {question}. Answer: Let's think step by step."
)

#question = input("Q: ")
question = "What is the capital of Sweden? "

repo_id = "google/flan-t5-xxl"
#repo_id = "google/mt5-base"

llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0.5, "max_length": 64})

llm_chain = LLMChain(prompt=prompt, llm=llm)

print(llm_chain.run(question))
