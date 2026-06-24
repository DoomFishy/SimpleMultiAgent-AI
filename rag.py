from rag_database import RagDatabase
import sys, math, json
from pypdf import PdfReader
import ollama

class RagAI:

    document = ""
    size = 1
    overlap = 0.1

    chunks = []

    database = None

    def __init__(self, document: str, size: int, overlap: float):

        self.document = document
        self.size = size
        self.overlap = overlap
        self.database = RagDatabase()


    def saveChunkEmbeddings(self, chunks, embeddings, filename="embeddings_cache.json"):
        
        self.database.storeRagChunks(chunks, embeddings)

    def loadChunkEmbeddings(self, filename="embeddings_cache.json"):

        return self.database.loadRagChunks()

    def extractTextFromPdf(self, pdf):
        reader = PdfReader(pdf)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text())

        return " ".join(full_text)

    def extractTextToChunk(self, text, chunk_size, overlap):
        size = chunk_size
        words = text.split()

        if len(words) < size:
            size = len(words)

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

    def searchForSimilarChunk(self, chunk_embed, question_embed, top_n):

        similarities = []
        top_indices = []

        for embed in chunk_embed:
            similarity = self.consineSimilarity(embed, question_embed[0])
            similarities.append(similarity)

        sorted_similarities = similarities
        sorted_similarities.sort(reverse=True)

        for i in range(0, len(similarities)):
            for j in range(0, len(sorted_similarities)):
                if similarities[i] == sorted_similarities[j]:
                    top_indices.append(i)

        return top_indices[0:top_n]

    def createChunkEmbedding(self, chunk_array):
        chunk_embedding = ollama.embed(
            model="nomic-embed-text",
            input=chunk_array
        )

        all_chunk_embeddings = chunk_embedding["embeddings"]

        return all_chunk_embeddings

    def createQuestionEmbedding(self, question, chunk_embed):
        question_embedding = ollama.embed(
            model="nomic-embed-text",
            input=question
        )

        all_question_embeddings = question_embedding["embeddings"]

        top_index = self.searchForSimilarChunk(chunk_embed, all_question_embeddings, top_n=3)

        return top_index

    def chat(self):
        text_string = self.extractTextFromPdf(self.document)
        chunks = self.extractTextToChunk(text_string, self.size, self.overlap)

        if self.database.checkNewChunk(chunks):

            chunk_embedding = self.createChunkEmbedding(chunks)
            self.saveChunkEmbeddings(chunks, chunk_embedding)
        
        chunks, chunk_embedding = self.loadChunkEmbeddings()


        messages = []

        print("Chat started! Type 'quit' to exit.\n")

        while True:

            user_question = input("User: ")
            print("\n")

            if user_question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break


            messages.append({"role": "user", "content": user_question})

        
            top_indices = self.createQuestionEmbedding(user_question, chunk_embedding)    

            parts = []

            for embed in top_indices:
                parts.append(chunks[embed])

            joined_parts = " ".join(parts)

            augmented_question = (f"Using this data: {joined_parts}. Respond to this question: {user_question}")

            responses = ollama.chat(
                model="gemma3",
                messages=messages + [{"role": "user", "content": augmented_question}],
                stream=True
            )

            print("AI: ")

            full_response = ""

            for chunk in responses:
                print(chunk["message"]["content"], end="", flush=True)
                full_response += chunk["message"]["content"]
            print("\n")

            messages.append({"role": "assistant", "content": full_response})