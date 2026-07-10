import os
import numpy as np
import ollama

class CompilerAgent:
    rag_result = ""
    web_result = ""

    def __init__(self, rag_result, web_result):
        self.rag_result = rag_result
        self.web_result = web_result

    def compileResult(self, rag_result, web_result):
        content = ""

        for item in rag_result:
            if isinstance(item, dict):
                chunk = item["chunk"]
            else:
                chunk = item
            content += chunk

        for item in web_result:
            if isinstance(item, dict):
                chunk = item["chunk"]
            else:
                chunk = item
            content += chunk

        prompt = f"""
            You are a Compiler Agent. Make a response using the result.

            Do not output introductionary phrases or conclusionary phrases.

            RESULT: {content}.
        """

        responses = ollama.chat(
            model="gemma3",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        response = ""

        for chunk in responses:
            response += chunk["message"]["content"]

        return response

    def run(self):
        return self.compileResult(self.rag_result, self.web_result)
