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
                pdf_files.append({
                    "path": pdf_path,
                    "name": f
                })

        return pdf_files

    def hasTag(self, user_question):
        words = user_question.lower().split()

        for word in words:
            start_char = word[0]
            end_char = word[-1]

            if start_char == "[" and end_char == "]":
                return True

        return False

    def findTag(self, user_question):
        start = user_question.find("[")
        end = user_question.find("]")

        pdf_names_string = user_question[start + 1:end]

        pdf_names_array = pdf_names_string.split(",")
    
        return pdf_names_array

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
            You are a Writer Agent. Your job is to combine both Compiler's draft while using the Validator's critique to produce a final response.
            Do not output introductionary phrases or conclusionary phrases about the Validator.

            If Critique is empty then ignore the critique.
            1. If Validator says "APPROVED": Output the Researcher's draft EXACTLY as-is. Do not add, remove, or modify anything.
            2. If Validator says "REVISE": Apply the Validator's critique to fix the draft, then output the corrected version.
            3. If Validator says "NEUTRAL": Output the Researcher's draft EXACTLY as-is. Do not add, remove, or modify anything.

            Determine if the draft ACTUALLY answers the question.

            DRAFT: {draft}
            CRITIQUE: {critique}
            QUESTION: {user_question}
            
            1. If the draft contains NO INFORMATION related to the question, ask the user to clarify

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

    def findIntent(self, user_question):
        keywords = [
            "summarize", "summarise", "summary", "summarization", "summarisation",
            "overview", "recap", "synopsis", "abstract", "digest", "condense", 
            "abridge", "abbreviated", "compressed", "notes",

            "keypoints", "key points", "key takeaways", "key takeaway",
            "main points", "main ideas", "main takeaways", "main takeaway",
            "important points", "important ideas", "highlights", "bullet points",
            "core ideas", "essential points", "major points", "critical points",
            
            "outline", "structure", "breakdown", "sections", "chapters",
            "table of contents", "contents", "index",
            
            "brief", "short", "shorten", "shortened", "shorter version",
            "quick", "quick version", "fast", "rapid",
            "tl;dr", "tldr", "tldr", "too long didn't read",
            
            "executive summary", "management summary", "high-level",
            "high level", "bird's eye view", "birds eye view",
            "big picture", "helicopter view", "30,000 foot", "thirty thousand foot",
            
            "gist", "essence", "core", "crux", "heart", "substance",
            "nub", "meat", "nutshell", "in a nutshell",
            
            "recap", "review", "wrap-up", "wrap up", "roundup", "round-up",
            "retrospective", "postmortem", "debrief", "debriefing",
            
            "explain", "explanation", "describe", "description",
            "tell me about", "what's in", "what is in",
            "what does it say", "what does this say",
            "what's the story", "what is the story",
        ]
        
        prompt = f"""
            Your job is to determine how the system should retrieve content to answer the user's question.
            Classify the question into ONE of these categories:


            1. "full_document" - The user wants an OVERVIEW, SUMMARY, or GENERAL UNDERSTANDING of the ENTIRE document.

            2. "specific_parts" - The user wants a SPECIFIC FACT, DETAIL, or ANSWER that exists in ONE part of the document.

            User question: {user_question}

            Output ONLY: "full_document" or "specific_parts"
        """

        responses = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}]
        )

        return responses['message']['content']

    def ask(self, user_question, rag_tool, web_tool):
        rag_result = ""
        web_result = ""

        user_intent = self.findIntent(user_question).strip()

        if user_intent == "full_document":
            rag_result = rag_tool.getPDF(user_question)
            web_result = ""
        else:
            rag_result = rag_tool.ask(user_question)
            web_result = web_tool.ask(user_question)

        print("Compiling...")
        compiler_agent = CompilerAgent(user_question, rag_result, web_result)
        draft = compiler_agent.run()

        print("Validating...")
        validator_agent = ValidatorAgent(draft, rag_result, web_result)
        critique = validator_agent.validate()

        print("Finalizing...")
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
        
            

    