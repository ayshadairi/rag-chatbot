from chat_interface import ChatInterface

if __name__ == "__main__":
    document_folder = "documents"

    bot = ChatInterface(document_folder)
    bot.chat()