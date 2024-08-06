import logging
from dotenv import load_dotenv
import os
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import SparseVector
from typing import List, Union
from sentence_transformers import CrossEncoder

from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent
from pydantic import BaseModel
from reranking_agent import ReRankingAgent


# Load environment variables
load_dotenv()
Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

#  Search Strategy Interface
class SearchStrategy:
    def search(self, query: str) -> List[str]:
        raise NotImplementedError

class SemanticSearch(SearchStrategy):
    def query_semantic_search(self, query: str) -> List[str]:
        # Load the dense embedding model
        embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # Initialize the Qdrant client
        qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

        # Embed the query using the dense embedding model
        dense_query = embedding_model.encode([query]).tolist()[0]

        # Perform the semantic search
        results = qdrant_client.search(
            collection_name=Collection_Name,
            query_vector=dense_query,
            limit=4,
        )

        return results

class HybridSearch(SearchStrategy):
    def query_hybrid_search(self, query: str) -> List[str]:

        embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY,
            timeout=30
        )

        # Embed the query using the dense embedding model
        dense_query = list(embedding_model.embed([query]))[0].tolist()

        # Embed the query using the sparse embedding model
        sparse_query = list(sparse_embedding_model.embed([query]))[0]

        results = qdrant_client.query_points(
            collection_name=Collection_Name,
            prefetch=[
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_query.indices.tolist(), values=sparse_query.values.tolist()),
                    using="sparse",
                    limit=4,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    limit=4,
                ),
            ],
            
            query=models.FusionQuery(fusion=models.Fusion.RRF), #Reciprocal Rerank Fusion
        )
        
        # Extract the text from the payload of each scored point
        documents = [point.payload['text'] for point in results.points]

        return documents

'''
def metadata_filter(file_names: Union[str, List[str]]) -> models.Filter:
    
    if isinstance(file_names, str):
        
        file_name_condition = models.FieldCondition(
            key="file_name",
            match=models.MatchValue(value=file_names)
        )
    else:
        
        file_name_condition = models.FieldCondition(
            key="file_name",
            match=models.MatchAny(any=file_names)
        )

    return models.Filter(
        must=[file_name_condition]
    )
'''

# Factory Function to Get the Appropriate Search Strategy
def get_search_strategy(search_type: str) -> SearchStrategy:
    if search_type == 'semantic':
        return SemanticSearch()
    elif search_type == 'hybrid':
        return HybridSearch()
    else:
        raise ValueError("Invalid search type")

class Retriever:
    def __init__(self, state: dict):
        self.state = state

    def retriever(self, search_type: str, query: str, reranking_model: str):
        """
        Perform the search and retrieval process based on the specified search type, query, and reranking model.
        """
        print("Starting the search and retrieval process")
        search_strategy = get_search_strategy(search_type)
        documents = search_strategy.query_hybrid_search(query)
        print("Search and retrieval process completed")
        reranked_documents = ReRankingAgent(query, documents, reranking_model)
        print("Reranking of the retrieved documents is complete")

        return reranked_documents

# RetrieverAgent function
def RetrieverAgent(state: dict) -> OpenAIAgent:


    def done() -> None:
        """
        Signal that the retrieval process is complete and update the state.
        """
        logging.info("Retrieval process is complete and updating the state")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    retriever = Retriever(state)

    tools = [
        FunctionTool.from_defaults(fn=retriever.retriever),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
    You are a helpful assistant that is performing search and retrieval tasks for a retrieval-augmented generation (RAG) system.
    Your task is to retrieve documents based on the user's query, search type, and reranking model.
    To do this, you need to know the search type, query, and reranking model.
    You can ask the user to supply these details.
    If the user supplies the necessary information, then call the tool "retriever" using the provided details to perform the search and retrieval process.
    The current user state is:
    {pprint.pformat(state, indent=4)}
    When you have completed the retrieval process, call the tool "done" to signal that you are done.
    If the user asks to do anything other than retrieve documents, call the tool "done" to signal that some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        system_prompt=system_prompt,
    )

if __name__ == '__main__':
    state = {}
    agent = RetrieverAgent(state = state)
    response = agent.chat("I want to query what is self-RAG? with Hybrid search and following with CrossEncoder Reranking model")
    print(response)