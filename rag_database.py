import chromadb
from chromadb.utils import embedding_functions
import uuid

class RagDatabase:
    client = None
    collection = None

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = None

    def storeChunks(self, chunks: str, embeddings: str):
        if self.collection == None:
            self.collection = self.client.get_or_create_collection("rag_data")

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        self.collection.add(
            ids=[ids],
            documents=[chunks],
            embeddings=embeddings,
        )

    def loadChunks(self):
        try:
            if self.collection == None:
                self.collection = self.client.get_or_create_collection("rag_data")

            results = self.collection.get()

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


    