from sentence_transformers import SentenceTransformer, util
import PyPDF2
import docx
import os
import torch
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import Counter


class DocumentProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = None
        self.documents = []  # Initialize as empty list, not None
        self.document_chunks = []  # Initialize as empty list, not None
        self.chunk_embeddings = None

    def extract_keywords(self, texts):
        """Extract important keywords from documents"""
        try:
            tfidf_matrix = self.keyword_vectorizer.fit_transform(texts)
            feature_names = self.keyword_vectorizer.get_feature_names_out()
            return set(feature_names)
        except:
            return set()

    def chunk_text(self, text, chunk_size=400, overlap=50):
        """Split text into overlapping chunks at sentence boundaries"""
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep some context by overlapping
                overlap_sentences = current_chunk.split('. ')
                current_chunk = '. '.join(overlap_sentences[-2:]) if len(overlap_sentences) >= 2 else ""

            current_chunk += " " + sentence if current_chunk else sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def load_documents(self, folder_path):
        print(f"🔄 Loading documents from: {folder_path}")

        # Clear existing documents to avoid duplicates
        self.documents = []
        self.document_chunks = []
        self.document_embeddings = None
        self.chunk_embeddings = None

        # Debug: List files in folder
        files = os.listdir(folder_path)
        print(f"📂 Files in folder: {files}")

        for filename in files:
            filepath = os.path.join(folder_path, filename)
            text = ""
            print(f"📄 Processing: {filename}")

            try:
                if filename.endswith('.txt'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    print(f"✅ Loaded TXT: {filename}, chars: {len(text)}")

                elif filename.endswith('.pdf'):
                    with open(filepath, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
                    print(f"✅ Loaded PDF: {filename}, chars: {len(text)}")

                elif filename.endswith('.docx'):
                    doc = docx.Document(filepath)
                    text = " ".join([para.text for para in doc.paragraphs])
                    print(f"✅ Loaded DOCX: {filename}, chars: {len(text)}")

                # Chunk the document text
                if text.strip():
                    chunks = self.chunk_text(text)
                    self.document_chunks.extend(chunks)
                    self.documents.append(text)
                    print(f"✅ Added to processing: {filename}, chunks: {len(chunks)}")
                else:
                    print(f"⚠️ Skipped (empty): {filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
                continue

        if self.document_chunks:
            print("🧠 Creating embeddings...")
            self.chunk_embeddings = self.embedding_model.encode(self.document_chunks, convert_to_tensor=True)
            print(f"✅ Created embeddings for {len(self.document_chunks)} chunks")
        else:
            print("⚠️ No document chunks to process!")

    def find_relevant_document(self, query, top_k=3, similarity_threshold=0.3):
        # Check if we have any document chunks
        if not self.document_chunks or self.chunk_embeddings is None:
            print("⚠️ No documents loaded or embeddings created")
            return []

        if len(self.document_chunks) == 0:
            print("⚠️ Document chunks list is empty")
            return []

        query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.chunk_embeddings)[0]
        top_results = torch.topk(cos_scores, k=min(top_k * 2, len(cos_scores)))

        # Multi-stage relevance filtering
        relevant_results = []
        for score, idx in zip(top_results[0], top_results[1]):
            score_value = score.item()

            if score_value < similarity_threshold:
                continue

            document_text = self.document_chunks[idx].lower()
            query_terms = query.lower().split()

            # Semantic relevance check
            term_matches = sum(1 for term in query_terms
                               if len(term) > 3 and term in document_text)

            # Contextual relevance scoring
            contextual_score = score_value
            if term_matches > 0:
                contextual_score *= (1 + (term_matches * 0.1))

            if contextual_score > similarity_threshold:
                relevant_results.append((self.document_chunks[idx], contextual_score, idx))

        # Sort by combined relevance score
        relevant_results.sort(key=lambda x: x[1], reverse=True)

        print(f"Query: '{query}'")
        print(f"Top similarity scores: {[f'{score.item():.3f}' for score in top_results[0][:3]]}")
        print(f"Relevant after filtering: {len(relevant_results)}")

        return [(doc, score) for doc, score, idx in relevant_results[:top_k]]

    def add_single_document(self, filepath):
        """Process and add a single document without reloading everything"""
        try:
            filename = os.path.basename(filepath)
            text = ""

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

                # Initialize if this is the first document
                if self.document_chunks is None:
                    self.document_chunks = []
                if self.documents is None:
                    self.documents = []

                # Add to existing documents
                self.document_chunks.extend(chunks)
                self.documents.append(text)

                # Update embeddings
                if self.chunk_embeddings is None:
                    self.chunk_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
                else:
                    new_embeddings = self.embedding_model.encode(chunks, convert_to_tensor=True)
                    self.chunk_embeddings = torch.cat([self.chunk_embeddings, new_embeddings])

                print(f"✅ Added {filename}: {len(chunks)} chunks")
                return True

        except Exception as e:
            print(f"❌ Error adding {filename}: {str(e)}")
            return False

        return False