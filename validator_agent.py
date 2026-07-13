import os
import numpy as np
import ollama

class ValidatorAgent:
    draft = ""

    rag_result = ""
    web_result = ""

    def __init__(self, draft, rag_result, web_result):
        self.draft = draft

        self.rag_result = rag_result
        self.web_result = web_result

    def validate(self):

        if self.web_result == "":
            return ""
        prompt = f"""
            You are the Quality Control Agent. Your job is to compare the Researcher's draft against the original RAG and Web snippets.
            Step 1: Scan for direct factual conflicts, date mismatches, and source authority gaps.
            Step 2: If NO conflicts exist, respond with: "APPROVED".
            Step 3: If conflicts exist, respond with: "REVISE" and provide a bulleted list of actionable fixes. Do not rewrite it yourself."

            For each factual claim in the draft, score it from 0-100 based on:

            - Agreement between RAG and Web (0 = total conflict, 100 = identical).

            - Recency (subtract 10 points for every year older than 2 years).

            DRAFT: {self.draft}

            RAG RESULT: {self.rag_result}
            WEB RESULT: {self.web_result}
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
