from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import os
import torch


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = None
        self.documents = []
        self.document_chunks = []
        self.chunk_embeddings = None

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    def load_documents(self, folder_path):
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)
            text = ""

            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                elif filename.endswith('.pdf'):
                    with open(filepath, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                elif filename.endswith('.docx'):
                    doc = docx.Document(filepath)
                    text = " ".join([para.text for para in doc.paragraphs])

                if text.strip():
                    chunks = self.chunk_text(text)
                    self.document_chunks.extend(chunks)
                    self.documents.append(text)

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

        if self.document_chunks:
            self.chunk_embeddings = self.embedding_model.encode(self.document_chunks, convert_to_tensor=True)

    def find_relevant_document(self, query, top_k=3, similarity_threshold=0.3):
        if not self.document_chunks:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k, len(cos_scores)))

        relevant_results = []
        for score, idx in zip(top_results[0], top_results[1]):
            if score > similarity_threshold:
                relevant_results.append((self.document_chunks[idx], score.item()))

        return relevant_results