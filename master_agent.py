import os
import numpy as np
import ollama

from rag_tool import RagTool
from web_tool import WebTool
from compiler_agent import CompilerAgent
from validator_agent import ValidatorAgent

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
        
    def finalize(self, draft, critique, user_question):

        prompt = f"""
            You are a Writer Agent. Your job is to combine both Reseracher's draft while using the Validator's critique to produce a final response.

            1. If Validator says "APPROVED": Output the Researcher's draft EXACTLY as-is. Do not add, remove, or modify anything.
            2. If Validator says "REVISE": Apply the Validator's critique to fix the draft, then output the corrected version.
                        
            Determine if the draft ACTUALLY answers the question.

            DRAFT: {draft}
            CRITIQUE: {critique}
            QUESTION: {user_question}
            
            1. If the draft contains NO INFORMATION related to the question, respond with:
            "I couldn't find any information about {user_question}."

            2. If the draft contains information but it's clearly irrelevant
            "I couldn't find any information about {user_question}."

            3. ONLY if the draft directly answers the question, output the draft.
        """

        responses = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in responses:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")

    def ask(self, user_question, rag_tool, web_tool):

        rag_result = rag_tool.ask(user_question)
        web_result = web_tool.ask(user_question)

        compiler_agent = CompilerAgent(rag_result, web_result)
        draft = compiler_agent.run()

        validator_agent = ValidatorAgent(draft, rag_result, web_result)
        critique = validator_agent.validate()

        self.finalize(draft, critique, user_question)
        


    def chat(self):
        message = []
        pdf_files = []

        rag_tool = RagTool()
        web_tool = WebTool()
        
        if not self.checkDirectoryValidity(self.directory_path):
            print("Directory is not valid!")
            return

        print("Chat started! Type 'quit' to exit.\n")
        
        while True:

            if self.checkNewFilesInDirectory(self.directory_path, pdf_files):
                pdf_files = self.getDirectoryFiles(self.directory_path)
                rag_tool.readPDF(pdf_files)

            user_question = input("User: ")
            print("\n")

            if user_question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            self.ask(user_question, rag_tool, web_tool)
        
            

    