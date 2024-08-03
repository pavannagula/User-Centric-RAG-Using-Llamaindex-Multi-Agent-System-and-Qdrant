from sentence_transformers import CrossEncoder

# Define the reranker models
class SentenceTransformerRerank:
    def __init__(self, model, top_n):
        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query, documents):
        # Compute the similarity scores between the query and each document
        scores = self.model.predict([(query, doc) for doc in documents])

        # Sort the documents based on their similarity scores
        ranked_documents = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

        # Select the top documents
        top_documents = [doc for doc, score in ranked_documents[:self.top_n]]

        return top_documents

# Dictionary of reranker models
RERANKERS = {
    "WithoutReranker": None,
    "CrossEncoder": SentenceTransformerRerank(model='cross-encoder/ms-marco-MiniLM-L-6-v2', top_n=2),
    "bge-reranker-base": SentenceTransformerRerank(model="BAAI/bge-reranker-base", top_n=2),
    "bge-reranker-large": SentenceTransformerRerank(model="BAAI/bge-reranker-large", top_n=2)
}

# ReRankingAgent function
def ReRankingAgent(query, documents, reranking_model):
    # Get the reranker model based on user preference
    reranker = RERANKERS.get(reranking_model)

    if reranker is None:
        # If no reranker is specified, return the documents as is
        return documents

    # Perform reranking
    top_documents = reranker.rerank(query, documents)

    return top_documents

