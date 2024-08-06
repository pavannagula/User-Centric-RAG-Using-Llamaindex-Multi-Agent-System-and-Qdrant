from dotenv import load_dotenv
import os
import json
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import PointStruct, SparseVector
from tqdm import tqdm

from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent
from reranking_agent import ReRankingAgent

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environmental variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('collection_name')
qdrant_client = QdrantClient(
                            url=Qdrant_URL,
                            api_key=Qdrant_API_KEY)
        

def QdrantIndexingAgent(state: dict) -> OpenAIAgent:  
        
    def load_nodes():
        metadata = []
        documents = []
        payload_file = r'C:\Users\pavan\Desktop\Generative AI\RAG-Automation-Using-Llamaindex-Agents-and-Qdrant\data\nodes.json'

        try:
            with open(payload_file, 'r') as file:
                nodes = json.load(file)

            for node in nodes:
                metadata.append(node['metadata'])
                documents.append(node['text'])

            print(f"Loaded {len(nodes)} the nodes from JSON file")

        except Exception as e:
            logging.error(f"Error loading nodes from JSON file: {e}")
            raise

        return documents, metadata

    def client_collection():
        """
        Create a collection in Qdrant vector database.
        """
        
        if not qdrant_client.collection_exists(collection_name=Collection_Name): 
            qdrant_client.create_collection(
                collection_name= Collection_Name,
                vectors_config={
                        'dense': models.VectorParams(
                            size=384,
                            distance = models.Distance.COSINE,
                        )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(
                                index=models.SparseIndexParams(
                                on_disk=False,              
                            ),
                        )
                    }
            )
            
        print(f"Created collection '{Collection_Name}' in Qdrant vector database.")


    def create_sparse_vector(sparse_embedding_model, text):
        """
        Create a sparse vector from the text using SPLADE.
        """
        sparse_embedding_model = sparse_embedding_model
        # Generate the sparse vector using SPLADE model
        embeddings = list(sparse_embedding_model.embed([text]))[0]

        # Check if embeddings has indices and values attributes
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")

    def insert_documents(embedding_model, documents, metadata):
        points = []
        embedding_model = TextEmbedding(model_name=embedding_model)
        sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        for i, (doc, metadata) in enumerate(tqdm(zip(documents, metadata), total=len(documents))):
            # Generate both dense and sparse embeddings
            dense_embedding = list(embedding_model.embed([doc]))[0]
            sparse_vector = create_sparse_vector(sparse_embedding_model, doc)

            # Create PointStruct
            point = models.PointStruct(
                id=i,
                vector={
                    'dense': dense_embedding.tolist(),
                    'sparse': sparse_vector,
                },
                payload={
                    'text': doc,
                    **metadata  # Include all metadata
                }
            )
            points.append(point)

        # Upsert points
        qdrant_client.upsert(
            collection_name=Collection_Name,
            points=points
        )

        print(f"Upserted {len(points)} points with dense and sparse vectors into Qdrant vector database.")

    def indexing(embedding_model) -> None:
        """
        Index the documents into the Qdrant vector database.
        """
        print("Starting to load the nodes from JSON file")
        documents, metadata = load_nodes()
        client_collection()
        print("Creation of the Qdrant Collection is Done")
        insert_documents(embedding_model, documents, metadata)
        print("Indexing of the nodes is complete")
    
    

    def done() -> None:
        """When you inserted the vetors into the Qdrant Cluster, call this tool."""
        logging.info("Indexing of the nodes is complete and updating the state")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    tools = [
        FunctionTool.from_defaults(fn=indexing),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
    You are a helpful assistant that is indexing documents for a retrieval-augmented generation (RAG) system.
    Your task is to index the documents into a Qdrant cluster.
    To do this, you need to know the embedding model to use.
    You can ask the user to supply this.
    If the user supplies the embedding model, Then, call the tool "indexing" using the provided embedding model to index the documents into the Qdrant cluster.
    The current user state is:
    {pprint.pformat(state, indent=4)}
    When you have indexed the documents into the Qdrant cluster, call the tool "done" to signal that you are done.
    If the user asks to do anything other than index the documents, call the tool "done" to signal some other agent should help.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        system_prompt=system_prompt,
    )

if __name__ == '__main__':
    state = {}
    agent = QdrantIndexingAgent(state = state)
    response = agent.chat("I want to index the documents using the sentence-transformers/all-MiniLM-L6-v2 embedding model.")