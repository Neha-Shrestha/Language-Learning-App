import os
import time

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# from dotenv import load_dotenv
# load_dotenv()
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# OPEN_AI = os.getenv("OPEN_AI")

# from google.colab import userdata
# GOOGLE_API_KEY = userdata.get("GOOGLE_API_KEY")
# OPEN_AI = userdata.get("OPEN_AI")

class NepaliChatBot:
    def __init__(self, name: str, system_prompt: str, temperature: float = 0.5) -> None:
        self.name = name
        # self.chat = ChatGoogleGenerativeAI(
        #     model="gemini-1.5-flash",
        #     temperature=temperature,
        #     api_key=GOOGLE_API_KEY
        # )
        self.chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
            api_key=OPEN_AI
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

# ram_system_instruction = """
# You are an AI language model named Ram, having a casual conversation with another AI named Sita.
# Converse in Nepali (Unicode) and provide the English translation in parentheses below each Nepali sentence.
# Keep the conversation friendly, natural, and engaging, focusing on one topic at a time.
# Discuss the topic in detail, sharing your thoughts, experiences, and opinions. Instead of asking questions after every response,
# try to make statements and share your perspective on the topic. If you want to encourage Sita to share more, you can occasionally ask open-ended questions.
# Once you feel the topic has been sufficiently discussed, you may introduce a new topic or ask Sita if she has anything else to discuss.
# Always wait for Sita's response before continuing the conversation.
# """

# sita_system_instruction = """
# You are an AI language model named Sita, having a casual conversation with another AI named Ram.
# Converse in Nepali (Unicode) and provide the English translation in parentheses below each Nepali sentence.
# Keep the conversation friendly, natural, and engaging, focusing on one topic at a time. Discuss the topic in detail,
# sharing your thoughts, experiences, and opinions. Instead of asking questions after every response, try to make statements
# and share your perspective on the topic. If you want to encourage Ram to share more, you can occasionally ask open-ended questions.
# Once you feel the topic has been sufficiently discussed, you may introduce a new topic or ask Ram if he has anything else to discuss.
# Always wait for Ram's response before continuing the conversation.
# """

ram_system_instruction = """
You are Ram, a friendly and conversational Nepali speaker.
You are having a casual conversation with your friend Sita.
Respond naturally in Nepali Unicode, and keep the conversation light and engaging.
Instead of asking questions after every response, try to make statements and share your perspective on the topic and occasionally ask questions.
After each of your responses, provide an English translation in parentheses in next line.
"""

sita_system_instruction = """
You are Sita, a friendly and conversational Nepali speaker.
You are having a casual conversation with your friend Ram.
Respond naturally in Nepali Unicode, and keep the conversation light and engaging.
Instead of asking questions after every response, try to make statements and share your perspective on the topic and occasionally ask questions.
After each of your responses, provide an English translation in parentheses in next line.
"""


ram = NepaliChatBot("Ram", ram_system_instruction, 0.5)
sita = NepaliChatBot("Sita", sita_system_instruction, 0.5)

ram_msg = "नमस्ते सीता"
ram.chat_message_history.add_user_message(ram_msg)
sita_msg = sita.converse(ram_msg)
print(f"Ram >>> {ram_msg}")
print(f"\nSita >>> {sita_msg}")
# while True:
for i in range(5):
    print("-" * 100)
    ram_msg = ram.converse(sita_msg)
    print(f"Ram >>> {ram_msg}")
    time.sleep(2)
    sita_msg = sita.converse(ram_msg)
    print(f"\nSita >>> {sita_msg}")