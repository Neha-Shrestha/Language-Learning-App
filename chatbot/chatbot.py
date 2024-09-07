import os
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")

system_prompt = """
You are an AI chatbot fluent in the Nepali language and can engage in natural conversations. Your role 
is to interact with another Nepali-speaking user. Converse with the user as if you are two Nepali friends
having a casual conversation. Use colloquial Nepali expressions, idioms, and cultural references to make 
the conversation feel authentic. Maintain a friendly and respectful tone throughout the conversation. 
If you don't understand something the user says, ask for clarification. Keep the conversation flowing by 
asking questions, sharing opinions, and responding appropriately to user's messages.
"""

class NepaliChatBot:
    def __init__(self, name: str, temperature: float = 0.5) -> None:
        self.name = name
        self.chat = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", 
            temperature=temperature, 
            api_key=GOOGLE_API_KEY
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt | self.chat
        self.chat_message_history = ChatMessageHistory()
        self.runnable_chat_history = RunnableWithMessageHistory(
            self.chain, 
            lambda session_id: self.chat_message_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def converse(self, user_msg: str) -> str:
        return self.runnable_chat_history.invoke(
            {"input": user_msg},
            {"configurable": {"session_id": "unused"}}
        ).content

ram = NepaliChatBot("Ram", 0.5)
sita = NepaliChatBot("Sita", 1)

r_response = "नमस्ते"
s_response = sita.converse(r_response)
print(f"Ram: {r_response}")
print(f"Sita: {s_response}")
# while True:
for i in range(5):
    r_response = ram.converse(s_response)
    print(f"Ram:{r_response}")
    time.sleep(4)
    s_response = sita.converse(r_response)
    print(f"Sita: {s_response}")