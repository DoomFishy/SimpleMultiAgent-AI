import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Get all collection names
collections = client.list_collections()
print("Available collections:", [col.name for col in collections])

for i, col in enumerate(collections):
    print(f"{i}. {col.name}")

while True:
    options = input("Please input database: ")
    collection = None

    match options:
        case "0":
            collection = client.get_collection("web_data")
            print("---- WEB DATA ----")

        case "1":
            collection = client.get_collection("qna_cache")
            print("---- QNA Cache ----")

        case _:
            print("Does not match any pattern")




    results = collection.get(include=["embeddings", "documents", "metadatas"])
    print("IDs:", results.get("ids", []))
    print("Documents:", results.get("documents", []))
    print("Embeddings:", results.get("embeddings", []))
    #print("Metadatas:", results.get("metadatas", []))
    #print("URIs:", results.get("uris", []))


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
"""