import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Get all collection names
collections = client.list_collections()
print("Available collections:", [col.name for col in collections])

"""
print("---- WEB DATA ----")
# Or get collection names directly
collection = client.get_collection("web_data")

results = collection.get()
# View separately:
print("IDs:", results.get("ids", []))
print("Documents:", results.get("documents", []))
print("Embeddings:", results.get("embeddings", []))
print("Metadatas:", results.get("metadatas", []))
print("URIs:", results.get("uris", []))

"""

print("---- RAG DATA ----")

# Or get collection names directly
collection = client.get_collection("rag_data")

results = collection.get()
# View separately:
print("IDs:", results.get("ids", []))
print("Documents:", results.get("documents", []))
print("Embeddings:", results.get("embeddings", []))
print("Metadatas:", results.get("metadatas", []))
print("URIs:", results.get("uris", []))
