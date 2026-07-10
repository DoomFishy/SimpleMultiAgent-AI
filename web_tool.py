from web_database import WebDatabase
import sys, math, json
import requests
import ollama

class WebTool:
    url = "http://localhost:8080/search"
    max_search = 5

    size = 5
    overlap = 0.1

    threshold = 0.85

    chunks = []

    def __init__(self, url="http://localhost:8080/search", max_search=5, size=5, overlap=0.1, threshold=0.65):
        self.url = url
        self.max_search = max_search
        self.size = size
        self.overlap = overlap
        self.threshold = threshold

        self.database = WebDatabase()

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

        top_indices = sorted(
            range(len(similarities)), 
            key=lambda i: similarities[i], 
            reverse=True
        )[:top_n]
        
        return top_indices, similarities

    def nomicEmbed(self, target):
        try:
            embed = ollama.embed(
                model="nomic-embed-text",
                input=target
            )

            return embed
        except Exception as e:
            print(f"Unable to nomic embed: {e}")

    def getKeywordsInQuestion(self, user_question):
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "must", "of", "at", "by", "for", "with", "without", "about", "against",
            "between", "through", "during", "within", "upon", "towards", "etc",
            "what", "when", "where", "which", "who", "whom", "whose", "why",
            "how", "can", "could", "would", "should", "may", "might", "must"
        }

        words = user_question.lower().split()

        keywords = []

        for word in words:
            if word not in stopwords:
                keywords.append(word)
        
        return keywords

    def compareQuestionToChunk(self, keywords, chunks):
        keyword_embeddings = self.nomicEmbed(keywords)


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

    def searchPastResponse(self, new_question, threshold=0.85):
        
        collection = self.database.client.get_or_create_collection("qna_cache")
        if collection is None:
            print("findPastResponse: Collection is empty!")
            return False
        
        similar_chunks = []

        document, all_question_embedding = self.database.loadQNACache()
        new_question_embedding = self.nomicEmbed(new_question)["embeddings"]

        top, similarities = self.findTopNSimilarEmbeddings(all_question_embedding, new_question_embedding, 5)

        for index in top:
            similarity = similarities[index]
            if similarity >= threshold:
                similar_chunks.append({
                    "chunk": document[index],
                    "similarity": similarities[index],
                    "confidence": "High"
                })

        if not len(similar_chunks) == 0:
            return similar_chunks
            
        return False

    def saveWebResponses(self, user_question, response):
        response_in_chunks = []
    
        if isinstance(response, str):
            response_in_chunks = self.extractTextToChunk(response, self.size, self.overlap)
        else:
            for item in response:
                if isinstance(item, dict):
                    chunk = item["chunk"]
                else:
                    chunk = item
                response_in_chunks.append(chunk)

        question_embedding = self.nomicEmbed(user_question)["embeddings"][0]
        
        duplicated_question_embeddings = [question_embedding] * len(response_in_chunks)

        self.database.storeQNACache(duplicated_question_embeddings, response_in_chunks)

    def searchWebContent(self, user_question, threshold=0.7):
        chunks, embeddings = self.database.loadWebContent()
        #question_embeddings = self.nomicEmbed(user_question)["embeddings"]

        keywords = self.getKeywordsInQuestion(user_question)
        keywords_embeddings = self.nomicEmbed(str(keywords))["embeddings"]

        top_indices, similarities = self.findTopNSimilarEmbeddings(embeddings, keywords_embeddings, top_n=5)

        similar_chunks = []

        for index in top_indices:
            if similarities[index] >= threshold: #Confident
                similar_chunks.append({
                    "chunk": chunks[index],
                    "similarity": similarities[index],
                    "confidence": "High"
                })

        if not len(similar_chunks) == 0:
            print("searchWebContent: web content found!")
            return similar_chunks
        
        return False

    def convertWebToChunks(self, result):
        web_content = ""

        for result in result:
            web_content += (f"Title: {result['title']} Content: {result['content']} URL: {result['url']} \n")

        chunks = self.extractTextToChunk(web_content, self.size, self.overlap)

        return chunks
        
    def saveWebContent(self, result):
        chunks = self.convertWebToChunks(result)
        embeddings = self.nomicEmbed(chunks)["embeddings"]   

        self.database.storeWebContent(chunks, embeddings)           

    def ask(self, user_question):

        past_result = self.searchPastResponse(user_question)

        if past_result != False:
            return past_result
        
        web_chunks = self.searchWebContent(user_question)

        if not web_chunks:
            result = self.searchWeb(user_question)

            self.saveWebContent(result)
            result_chunks = self.convertWebToChunks(result)
            self.saveWebResponses(user_question, result_chunks)

            return result_chunks

        self.saveWebResponses(user_question, web_chunks)
        return web_chunks








