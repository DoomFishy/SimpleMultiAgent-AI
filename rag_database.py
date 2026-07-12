import chromadb
from chromadb.utils import embedding_functions
import uuid

class RagDatabase:
    client = None
    collection = None

    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("rag_content")


    def storeRagChunks(self, chunks, embeddings, name):
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]

        metadatas = [{
            "name": name
        }]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        #print(f"Added IDS: {ids} \n DOC: {chunks} \n Embed: {embeddings}")

    def loadRagChunks(self):
        try:
            results = self.collection.get(include=["documents", "embeddings"])

            chunks = results.get("documents", [])
            embeddings = results.get("embeddings", [])
         
            if embeddings is None:
                embeddings = []

            if not chunks:
                return None, None

            return chunks, embeddings

        except Exception as e:
            print(f"Error loading from ChromaDB: {e}")
            return None, None

    def checkNewChunk(self, chunks):
        collection = self.client.get_collection("rag_content")
        
        results = collection.get()
        existing_chunks = results.get("documents", [])
        
        for new_chunk in chunks:
            if new_chunk in existing_chunks:
                return False
        
        return True
    