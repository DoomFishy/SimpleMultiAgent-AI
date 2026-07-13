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

        prompt = f"""
            You are a Compiler Agent. Fulfill the user's request using ONLY the provided content.

            USER_QUESTION: {user_question}
            RAG_RESULT: {rag_content}
            WEB_RESULT: {web_content if web_content else "None provided"}

            GENERAL INSTRUCTIONS:
            1. Do what the user asks:
            - If they ask for notes → give structured notes with headers and bullets
            - If they ask for a summary → give a concise summary
            - If they ask a question → give a direct answer
            - If they ask for a list → give a clear list
            - If they ask for an explanation → give a clear explanation

            2. Adapt your response format to match the request:
            - Notes → use ## headers and bullet points
            - Summary → use paragraphs
            - Lists → use bullet points or numbers
            - Comparison → use a compare/contrast structure
            - Steps → use numbered steps

            3. Always:
            - Use ONLY the provided content
            - Start directly with your response
            - No introductory or concluding phrases
            - No "I'm an AI" or "based on the content"
            - Be clear, helpful, and complete

            RESPONSE:
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
