import chromadb
from chromadb.utils import embedding_functions
import uuid, math

class WebDatabase:
    client = None
    collection = None

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None

    def storeWebContent(self, chunks, embeddings):
        if self.collection == None:
            self.collection = self.client.get_or_create_collection("web_content")

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings
        )

    def loadWebContent(self):
        try:
            if self.collection == None:
                self.collection = self.client.get_or_create_collection("web_content")

            results = self.collection.get(include=["documents", "embeddings"])

            chunks = results.get("documents", [])
            embeddings = results.get("embeddings", [])

            if embeddings is None:
                embeddings = []

            if chunks is None:
                return None, None

            return chunks, embeddings

        except Exception as e:
            print(f"Error loading from ChromaDB: {e}")
            return None, None


    def storeQNACache(self, user_question_embedding, response):
        if self.collection == None:
            self.collection = self.client.get_or_create_collection("qna_cache")

        ids = [str(uuid.uuid4()) for _ in range(len(response))]

        self.collection.add(
            ids=ids,
            documents=response,
            embeddings=user_question_embedding
        )

    def loadQNACache(self):
        try:
            if self.collection == None:
                self.collection = self.client.get_or_create_collection("qna_cache")

            results = self.collection.get(include=["documents", "embeddings"])

            response = results.get("documents", [])
            question_embeddings = results.get("embeddings", [])

            if question_embeddings is None:
                question_embeddings = []

            if response is None:
                return None, None

            return response, question_embeddings

        except Exception as e:
            print(f"Error loading from ChromaDB: {e}")
            return None, None
