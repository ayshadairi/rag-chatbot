import google.generativeai as genai
from document_processor import DocumentProcessor
import os
import re


class GeminiChat:
    def __init__(self, document_folder, api_key=None):
        self.processor = DocumentProcessor()
        self.processor.load_documents(document_folder)

        # Configure Gemini
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

        self.model = genai.GenerativeModel('gemini-1.5-flash-latest')

    def generate_response(self, query):
        try:
            # First, find relevant documents using our RAG system
            relevant_docs = self.processor.find_relevant_document(query, top_k=3, similarity_threshold=0.3)

            if not relevant_docs:
                return "I couldn't find relevant information about that topic in my documents. Please ask about AI, software development, or related topics."

            # Prepare context from documents
            context = "\n".join([f"- {doc[0]}" for doc in relevant_docs[:2]])

            # Create intelligent prompt for Gemini
            prompt = f"""You are an intelligent assistant that answers questions based on provided documents.

DOCUMENT CONTEXT:
{context}

CONVERSATION HISTORY:
{" | ".join(self.conversation_history[-3:])}

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based primarily on the document context provided
2. Be helpful, concise, and informative
3. If the documents don't fully answer the question, say so politely
4. If the question is unrelated to the documents, explain your focus is on AI and software development
5. Use a natural, conversational tone

ANSWER:"""

            # Generate response using Gemini
            response = self.model.generate_content(prompt)

            # Clean up the response
            answer = response.text.strip()

            # Add to conversation history
            self.conversation_history.append(f"User: {query}")
            self.conversation_history.append(f"Assistant: {answer}")

            # Keep history manageable
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

            return answer

        except Exception as e:
            error_msg = str(e)
            if "API_KEY" in error_msg:
                return "API key error. Please check your Gemini API key."
            elif "quota" in error_msg.lower():
                return "API quota exceeded. Please try again later or check your Gemini usage."
            else:
                return f"I encountered an error: {error_msg}"


# For testing without web interface
if __name__ == "__main__":
    # Replace with your actual Gemini API key
    API_KEY = "AIzaSyDGJ8R2jU2lRRbcjN0rhvzUvTgK4v6DLwo"  # ← Replace this!

    if API_KEY == "AIzaSyDGJ8R2jU2lRRbcjN0rhvzUvTgK4v6DLwo":
        print("❌ Please replace API_KEY with your actual Gemini API key")
        print("💡 Get free key from: https://makersuite.google.com/")
    else:
        chat = GeminiChat("documents", api_key=API_KEY)

        print("\n🤖 Gemini RAG Chatbot Ready!")
        print("💡 Try questions like: 'What is GitHub Copilot?' or 'How is AI affecting developers?'")
        print("💬 Type 'quit' to exit\n")

        while True:
            query = input("You: ")
            if query.lower() in ['quit', 'exit', 'bye']:
                break
            response = chat.generate_response(query)
            print(f"\nAssistant: {response}\n")