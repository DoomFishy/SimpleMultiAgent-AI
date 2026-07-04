import chromadb
from chromadb.utils import embedding_functions
import uuid, math

class WebDatabase:
    client = None
    collection = None

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None

    def storeWebResponse(self, user_question_embedding, response):
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
