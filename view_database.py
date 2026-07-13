import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

# Get all collection names
collections = client.list_collections()
print("Available collections:", [col.name for col in collections])

for i, col in enumerate(collections):
    print(f"{i}. {col.name}")

while True:
    options = input("Please input database or enter delete: ")
    collection = None

    for i, col in enumerate(collections):
        if str(i) == options:
            collection = client.get_collection(col.name)
            print(f"----[ {col.name} ]----")

    if options == "delete":
        for col in collections:
            client.delete_collection(col.name)

    if not collection:
        print("Database not found!")
        continue

    results = collection.get(include=["embeddings", "documents", "metadatas"])
    print("IDs:", results.get("ids", []))
    print("Documents:", results.get("documents", []))
    print("Embeddings:", results.get("embeddings", []))
    print("Metadatas:", results.get("metadatas", []))
    #print("URIs:", results.get("uris", []))
