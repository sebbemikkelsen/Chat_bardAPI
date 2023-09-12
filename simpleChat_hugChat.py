from hugchat import hugchat
from hugchat.login import Login
import os
from dotenv import load_dotenv

load_dotenv()

class HC_bot:
    def __init__(self, email, pw) -> None:
        self.email = email
        self.pw = pw
        self.sign = Login(email, pw)
        self.cookies = self.sign.login()

    def new_chat(self):
        # Save cookies to the local directory
        cookie_path_dir = "./cookies_snapshot"
        self.sign.saveCookiesToDir(cookie_path_dir)

        # Create a ChatBot
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

        # Create a new conversation
        self.id = self.chatbot.new_conversation()
        #chatbot.change_conversation(id)

        self.chat()

    def chat(self):

        question = input("Q: ")
        while question != "quit":
            print("ANS: ", self.chatbot.chat(question))
            question = input("Q: ")

        # Get conversation list
        #conversation_list = chatbot.get_conversation_list()

    def get_summary(self):
        print(self.chat.summarize_conversation(self.id))


    def switch_model(self, model):
        if model == "OpenAssistant":
            self.chatbot.switch_llm(0) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
        elif model == "meta-llama":
            self.chatbot.switch_llm(1) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
        elif model == "codellama":
            self.chatbot.switch_llm(2) #'codellama/CodeLlama-34b-Instruct-hf'
        else:
            print("error, choose between: OpenAssistant, meta-llama, or codellama")








def main():
    email = os.getenv("EMAIL_HF")
    pw = os.getenv("PASS_HF")
    bot = HC_bot(email, pw)

    bot.new_chat()




if __name__ == '__main__':
    main()


