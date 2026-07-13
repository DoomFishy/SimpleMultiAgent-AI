import os
import numpy as np
import ollama

class CompilerAgent:
    rag_result = ""
    web_result = ""
    user_question = ""

    def __init__(self, user_question, rag_result, web_result):
        self.rag_result = rag_result
        self.web_result = web_result
        self.user_question

    def compileResult(self, user_question, rag_result, web_result):
        rag_content = ""
        web_content = ""

        for item in rag_result:
            if isinstance(item, dict):
                chunk = item["chunk"]
            else:
                chunk = item
            rag_content += chunk

        for item in web_result:
            if isinstance(item, dict):
                chunk = item["chunk"]
            else:
                chunk = item
            web_content += chunk

        print(rag_content)

        prompt = f"""
            You are a Compiler Agent. Make a response using the result based off of the user question.
            If web_result is empty, just use rag_result.
            Do not output introductionary phrases or conclusionary phrases.

            USER_QUESTION: {user_question}.
            RAG_RESULT: {rag_content}.
            WEB_RESULT: {web_content}.

        """

        responses = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}]
        )

        responses = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}]
        )

        return responses['message']['content']

    def run(self):
        return self.compileResult(self.user_question, self.rag_result, self.web_result)
