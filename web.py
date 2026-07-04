from web_database import WebDatabase
import sys, math, json
import requests
import ollama

class WebAI:
    url = "http://localhost:8080/search"
    max_search = 5

    size = 5
    overlap = 0.1

    chunks = []

    def __init__(self, url: str, max_search: int, size: int, overlap: float):
        self.url = url
        self.max_search = max_search

        self.size = size
        self.overlap = overlap

        self.database = WebDatabase()

    def saveChunkEmbeddings(self, chunks, embeddings,):
        print("saveChunkEmbeddings: Saving chunk embeddings")

        self.database.storeWebChunks(chunks, embeddings)

    def loadChunkEmbeddings(self, filename="embeddings_cache.json"):
        return self.database.loadWebChunks()

    def extractTextToChunk(self, text, chunk_size, overlap):
        size = chunk_size
        words = text.split()

        if len(words) < size:
            size = len(words)

        words_per_chunk = len(words) // size

        chunk_arr = []
        min_range = 0
        max_range = words_per_chunk
        overlap_words = int(words_per_chunk * overlap)

        for i in range(0, size):
            joinedWords = " ".join(words[min_range:max_range])

            chunk_arr.append(joinedWords)


            min_range = max_range - overlap_words
            max_range += words_per_chunk

        return chunk_arr

    def consineSimilarity(self, vector_1, vector_2):
        dot_product = sum(a * b for a, b in zip(vector_1, vector_2))

        magnitude_1 = math.sqrt(sum(a * a for a in vector_1))
        magnitude_2 = math.sqrt(sum(b * b for b in vector_2))

        return dot_product / (magnitude_1 * magnitude_2)

    def findTopNSimilarEmbeddings(self, chunk_embed, question_embed, top_n):

        similarities = []
        top_indices = []

        for embed in chunk_embed:
            similarity = self.consineSimilarity(embed, question_embed[0])
            similarities.append(similarity)

        sorted_similarities = similarities
        sorted_similarities.sort(reverse=True)

        for i in range(0, len(similarities)):
            for j in range(0, len(sorted_similarities)):
                if similarities[i] == sorted_similarities[j]:
                    top_indices.append(i)

        return top_indices[0:top_n], sorted_similarities

    def nomicEmbed(self, array):
        try:
            embed = ollama.embed(
                model="nomic-embed-text",
                input=array
            )

            return embed
        except Exception as e:
            print(f"Unable to nomic embed: {e}")

    def searchWeb(self, question):
        params = {
            "q": question,
            "format": "json"
        }

        try:
            response = requests.get(self.url, params=params)

            web_data = response.json()

            formatted_results = []

            for result in web_data["results"][:self.max_search]:
                formatted_results.append({
                    "title": result["title"],
                    "content": result["content"],
                    "url": result["url"]
                })

            return formatted_results

        except requests.exceptions.RequestException as e:
            print(f"Search Failed: {e}")

    def findPastResponse(self, new_question, threshold=0.85):
        
        collection = self.database.collection
        if collection is None:
            print("No collection found!")
            return False
        
        response = ""

        document, all_question_embedding = self.database.loadQNACache()
        new_question_embedding = self.nomicEmbed(new_question)["embeddings"]

        top, similarities = self.findTopNSimilarEmbeddings(all_question_embedding, new_question_embedding, 3)

        for index in top:
            if similarities[index] >= threshold:
                response += document[index]

        if response != "":
            print("findPastResponse: response found!")
            return response

            
        print("findPastResponse: past response cannot be found!")
        return False

    def repeatPastResponse(self, user_question, past_response):
        try:
            stream = ollama.chat(
                model="gemma3",
                messages=[{"role": "user", "content": f"Using your past response {past_response} answer {user_question}"}],
                stream=True
            )

            print("AI: ")

            for chunk in stream:
                print(chunk["message"]["content"], end="", flush=True)
            print("\n") 
        
        except:
            print("Error: problem with repeatPastResponse")

    def saveWebResponses(self, user_question, response):
        response_in_chunks = self.extractTextToChunk(response, self.size, self.overlap)
        question_embedding = self.nomicEmbed(user_question)["embeddings"][0]
        
        duplicated_question_embeddings = [question_embedding] * len(response_in_chunks)

        self.database.storeWebResponse(duplicated_question_embeddings, response_in_chunks)
        print("Stored!")

    def askAI(self, user_question, search_results, total_message):

        web_content = ""
        full_response = ""

        for result in search_results:
            web_content += (f"Title: {result['title']} Content: {result['content']} URL: {result['url']} \n")

        stream = ollama.chat(
            model="gemma3",
            messages=total_message+[{"role": "user", "content": f"Using the search results {web_content} answer the question {user_question}"}],
            stream=True
        )

        print("AI: ")

        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            full_response += chunk["message"]["content"]
        print("\n") 

        self.saveWebResponses(user_question, full_response)

        return full_response

    def chat(self):
        messages = []

        print("Chat started! Type 'quit' to exit.\n")

        while True:
            user_question = input("User: ")
            print("\n")

            if user_question.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            response, question_embeddings = self.database.loadQNACache()
            past_result = self.findPastResponse(user_question)

            if past_result != False:
                print("Question inputted is a duplicate!")
                self.repeatPastResponse(user_question, past_result)

            else:
                print("New question found!")

                result = self.searchWeb(user_question)

                messages.append({"role": "user", "content": user_question})

                response = self.askAI(user_question, result, messages)

                messages.append({"role": "assistant", "content": response})


        """
        question_chunk_embedding = self.createChunkEmbedding(chunks)
        top_indices = self.searchForSimilarChunk(chunk_embedding, question_chunk_embedding, top_n=3)
        
        parts = []

        for embed in top_indices:
            parts.append(chunks[embed])

        joined_top_chunks = " ".join(parts)

        joined_web_results = ""

        urls = []

        for i, result in enumerate(search_results, 1):
            joined_web_results += 
            Source {i}:
            Title: {result['title']}
            URL: {result['url']}
            Content: {result['content']}
           

            urls.append(result['url'])

        combined_text =
        Retrieved knowledge from your database: {joined_top_chunks}
        Recent Web result: {joined_web_results}
        Based on both sources answer the given question {question}
        if there is conflicting information, prioritize the most recent source
        
        """