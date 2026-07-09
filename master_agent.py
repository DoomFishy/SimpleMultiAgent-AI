import os
import numpy as np
import ollama

from rag_agent import RagAgent
from web_agent import WebAgent

class MasterAgent:
    directory_path = ""
    size = 5
    overlap = 0.1
    threshold = 0.85
    search_limit = 5

    def __init__(self, directory_path, size=5, overlap=0.1, threshold=0.85, search_limit=5):
        self.directory_path = directory_path
        self.size = size
        self.overlap = overlap
        self.threshold = threshold
        self.search_limit = search_limit

    def checkDirectoryValidity(self, path):
        if not os.path.exists(path):
            return False, "Directory does not exist!"
        if not os.path.isdir(path):
            return False, "Path is not a directory!"
        
        pdf_files = self.getDirectoryFiles(path)

        if not pdf_files:
            return False, "Directory contains no PDF files"
        
        return True
    
    def getDirectoryFiles(self, path):
        pdf_files = []

        for f in os.listdir(path):
            if f.lower().endswith(".pdf"):
                pdf_path = f"{path}/{f}"
                pdf_files.append(pdf_path)

        return pdf_files

    def checkNewFilesInDirectory(self, path, pdf_files):
        new_pdf_files = self.getDirectoryFiles(path)

        new_pdf_arr = np.array(new_pdf_files)
        pdf_arr = np.array(pdf_files)

        if pdf_arr.shape != new_pdf_arr.shape:
            return True
        
        else:
            return False

    def askAI(self, user_question, rag_agent, web_agent, pdf_files):

        rag_result = rag_agent.ask(user_question)
        web_result = web_agent.ask(user_question)

        if rag_result:
            print("RAG: I CAN ANSWER THIS!")
        else:
            print(f"WEB RESULT: {web_result}")


    def chat(self):
        message = []
        pdf_files = []

        rag_agent = RagAgent()
        web_agent = WebAgent()
        
        if not self.checkDirectoryValidity(self.directory_path):
            print("Directory is not valid!")
            return

        print("Chat started! Type 'quit' to exit.\n")
        
        while True:

            if self.checkNewFilesInDirectory(self.directory_path, pdf_files):
                pdf_files = self.getDirectoryFiles(self.directory_path)
                rag_agent.readPDF(pdf_files)
                print("New PDF Detected!")

            user_question = input("User: ")
            print("\n")

            if user_question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            self.askAI(user_question, rag_agent, web_agent, pdf_files)
        
            

    