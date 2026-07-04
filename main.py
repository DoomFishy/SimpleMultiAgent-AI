from rag import RagAI
from web import WebAI


def rag():
    print("--- RUNNING RAG! ---")
    rag_ai = RagAI("test.pdf", 5, 0.1)
    rag_ai.chat()

def web():
    print("--- RUNNING WEB! ---")

    web_ai = WebAI("http://localhost:8080/search", 5, 5, 0.1)
    web_ai.chat()


web()