from rag_database import RagDatabase
import sys, math, json
from pypdf import PdfReader
import ollama

class RagTool:

    size = 5
    overlap = 0.1
    threshold = 0.5

    chunks = []

    database = None

    def __init__(self, size=5, overlap=0.1, threshold=0.7):

        self.size = size
        self.overlap = overlap
        self.threshold = threshold
        self.database = RagDatabase()

    def saveChunkEmbeddings(self, chunks, embeddings):
        
        self.database.storeRagChunks(chunks, embeddings)

    def loadChunkEmbeddings(self):

        return self.database.loadRagChunks()

    def extractTextFromPDF(self, pdf):
        reader = PdfReader(pdf)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text())

        return " ".join(full_text)

    def extractTextToChunk(self, text, chunk_size, overlap):
        size = chunk_size
        words = text.split()

        if len(words) < size:
            size = 1

        words_per_chunk = len(words) // size

        chunk_arr = []
        min_range = 0
        max_range = words_per_chunk
        overlap_words = int(words_per_chunk * overlap)

        for i in range(0, size):
            joinedWords = " ".join(words[min_range:max_range])

            chunk_arr.append(joinedWords)


            min_range = max_range - overlap_words
            max_range += words_per_chunk

        return chunk_arr

    def consineSimilarity(self, vector_1, vector_2):
        dot_product = sum(a * b for a, b in zip(vector_1, vector_2))

        magnitude_1 = math.sqrt(sum(a * a for a in vector_1))
        magnitude_2 = math.sqrt(sum(b * b for b in vector_2))

        return dot_product / (magnitude_1 * magnitude_2)

    def findTopNSimilarEmbeddings(self, chunk_embed, question_embed, top_n):

        similarities = []
        top_indices = []

        for embed in chunk_embed:
            similarity = self.consineSimilarity(embed, question_embed[0])
            similarities.append(similarity)

        if top_n == 0:
            top_indices = sorted(
                range(len(similarities)), 
                key=lambda i: similarities[i], 
                reverse=True
            )
            return top_indices, similarities

        else:
            top_indices = sorted(
                range(len(similarities)), 
                key=lambda i: similarities[i], 
                reverse=True
            )[:top_n]

            return top_indices[0:top_n], similarities

    def nomicEmbed(self, target):
        embedding = ollama.embed(
            model="nomic-embed-text",
            input=target
        )

        all_embeddings = embedding["embeddings"]

        return all_embeddings            

    def readPDF(self, pdf_files):

        for pdf in pdf_files:

            text = self.extractTextFromPDF(pdf["path"])
            chunks = self.extractTextToChunk(text, self.size, self.overlap)

            if self.database.checkNewChunk(chunks):
                chunk_embedding = self.nomicEmbed(chunks)
                self.saveChunkEmbeddings(chunks, chunk_embedding, pdf["name"])
    
    def getPDF(self, user_question, pdf_files):
        pdf_names = [pdf["name"] for pdf in pdf_files]
        name_embeddings = self.nomicEmbed(pdf_names)
        question_embeddings = self.nomicEmbed(user_question)

        top_indices, similarities = self.findTopNSimilarEmbeddings(name_embeddings, question_embeddings, top_n=0)        

        for i in pdf_files:
            print(i["name"])

    def ask(self, user_question):
        
        chunks, chunk_embeddings = self.loadChunkEmbeddings()

        question_embeddings = self.nomicEmbed(user_question)    
        top_indices, similarities = self.findTopNSimilarEmbeddings(chunk_embeddings, question_embeddings, top_n=3)        

        similar_chunks = []

        for index in top_indices:
            if similarities[index] >= self.threshold: #Confident
                similar_chunks.append({
                    "chunk": chunks[index],
                    "similarity": similarities[index],
                    "confidence": "High"
                })

        return similar_chunks
