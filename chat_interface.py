from document_processor import DocumentProcessor
import re
import random
import time


class ChatInterface:
    def __init__(self, document_folder=None):
        self.processor = DocumentProcessor()
        self.conversation_history = []

        if document_folder:
            self.processor.load_documents(document_folder)

    def understand_query(self, query):
        """Simple query understanding"""
        query_lower = query.lower()

        # Check if query is related to our documents
        ai_terms = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'neural network']
        dev_terms = ['develop', 'code', 'programming', 'software', 'github', 'copilot', 'debug']

        is_related = any(term in query_lower for term in ai_terms + dev_terms)

        return {
            'is_related': is_related,
            'is_question': query_lower.startswith(('what', 'how', 'why', 'when', 'where', 'who'))
        }

    def generate_response(self, query):
        print(f"Processing: '{query}'")

        # Simple query understanding
        query_info = self.understand_query(query)

        if not query_info['is_related']:
            return "I specialize in AI and software development topics. Could you ask about those areas?"

        # Find relevant content
        relevant_docs = self.processor.find_relevant_document(query, top_k=2, similarity_threshold=0.3)

        if not relevant_docs:
            return "I couldn't find specific information about that in my documents. Try asking about AI tools or software development."

        # Use the most relevant document
        best_doc, best_score = max(relevant_docs, key=lambda x: x[1])
        print(f"Best score: {best_score:.3f}")

        # Simple response generation
        response_templates = [
            "Based on the documents: {}",
            "I found this information: {}",
            "According to the sources: {}",
            "Here's what I learned: {}"
        ]

        # Clean up the response
        sentences = re.split(r'(?<=[.!?]) +', best_doc)
        if sentences:
            clean_response = sentences[0]  # Use first complete sentence
            if len(sentences) > 1:
                clean_response += " " + sentences[1]  # Add second sentence if available
        else:
            clean_response = best_doc[:200] + "..."  # Fallback

        response = random.choice(response_templates).format(clean_response)
        return response

    def chat(self):
        """Interactive chat interface"""
        print("RAG Chatbot initialized. Type 'quit' to exit.")
        print(f"Loaded {len(self.processor.documents)} documents.")
        while True:
            query = input("\nYou: ")
            if query.lower() in ['quit', 'exit']:
                break
            response = self.generate_response(query)
            print(f"Bot: {response}")