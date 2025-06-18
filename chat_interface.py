# File: chat_interface.py
from transformers import AutoTokenizer, AutoModel
from document_processor import DocumentProcessor

class ChatInterface:
    def __init__(self, document_folder=None):
        self.processor = DocumentProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.lm_model = AutoModel.from_pretrained("distilgpt2")
        if document_folder:
            self.processor.load_documents(document_folder)

    def generate_response(self, query):
        relevant_docs = self.processor.find_relevant_document(query)
        if not relevant_docs:
            return "I couldn't find any relevant information in the provided documents."
        context = relevant_docs[0][0]
        input_text = f"Based on the following information: {context[:1000]}\n\nQuestion: {query}\nAnswer:"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.lm_model.generate(**inputs, max_length=200)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "Answer:" in response:
            response = response.split("Answer:")[1].strip()
        return response

    def chat(self):
        """Interactive chat interface"""
        print("Private RAG Chatbot initialized. Type 'quit' to exit.")
        print(f"Loaded {len(self.processor.documents)} documents.")
        while True:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
            response = self.generate_response(query)
            print(f"Bot: {response}")