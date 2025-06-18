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

    def load_documents(self, folder_path):
        for filename in os.listdir(folder_path):
            filepath = os.path.join(folder_path, filename)

            if filename.endswith('.txt'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.documents.append(text)

            elif filename.endswith('.pdf'):
                with open(filepath, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    text = " ".join([page.extract_text() for page in reader.pages])
                self.documents.append(text)

            elif filename.endswith('.docx'):
                doc = docx.Document(filepath)
                text = " ".join([para.text for para in doc.paragraphs])
                self.documents.append(text)

        if self.documents:
            self.document_embeddings = self.embedding_model.encode(self.documents, convert_to_tensor=True)

    def find_relevant_document(self, query, top_k=3):
        if not self.documents:
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.document_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        return [(self.documents[idx], score.item()) for score, idx in zip(top_results[0], top_results[1])]