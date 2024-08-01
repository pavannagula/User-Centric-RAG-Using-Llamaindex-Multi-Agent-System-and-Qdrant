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

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environmental variables from a .env file
load_dotenv()

Qdrant_API_KEY = os.getenv('Qdrant_API_KEY')
Qdrant_URL = os.getenv('Qdrant_URL')
Collection_Name = os.getenv('Collection_Name')

class QdrantIndexing:
    """
    A class for indexing documents using Qdrant vector database.
    """

    def __init__(self, state, embedding_model) -> None:
        """
        Initialize the QdrantIndexing object.
        """
        self.state = state
        self.openai_agent = OpenAIAgent()
        self.data_path = r"C:\Users\pavan\Desktop\Generative AI\RAG-Automation-Using-Llamaindex-Agents-and-Qdrant\data\nodes.json"
        self.embedding_model = TextEmbedding(model_name=embedding_model)
        self.sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.qdrant_client = QdrantClient(
                            url=Qdrant_URL,
                            api_key=Qdrant_API_KEY)
        self.metadata = []
        self.documents = []
        logging.info("QdrantIndexing object initialized.")

    def load_nodes(self) -> None:
        """
        Load nodes from a JSON file and extract metadata and documents.
        """
        with open(self.data_path, 'r') as file:
            nodes = json.load(file)

        for node in nodes:
            self.metadata.append(node['metadata'])
            self.documents.append(node['text'])

        logging.info(f"Loaded {len(nodes)} nodes from JSON file.")

    def client_collection(self):
        """
        Create a collection in Qdrant vector database.
        """
        if not self.qdrant_client.collection_exists(collection_name=f"{Collection_Name}"): 
            self.qdrant_client.create_collection(
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
            logging.info(f"Created collection '{Collection_Name}' in Qdrant vector database.")

    def create_sparse_vector(self, text):
        """
        Create a sparse vector from the text using SPLADE.
        """
        # Generate the sparse vector using SPLADE model
        embeddings = list(self.sparse_embedding_model.embed([text]))[0]

        # Check if embeddings has indices and values attributes
        if hasattr(embeddings, 'indices') and hasattr(embeddings, 'values'):
            sparse_vector = models.SparseVector(
                indices=embeddings.indices.tolist(),
                values=embeddings.values.tolist()
            )
            return sparse_vector
        else:
            raise ValueError("The embeddings object does not have 'indices' and 'values' attributes.")


    def insert_documents(self):
        points = []
        for i, (doc, metadata) in enumerate(tqdm(zip(self.documents, self.metadata), total=len(self.documents))):
            # Generate both dense and sparse embeddings
            dense_embedding = list(self.embedding_model.embed([doc]))[0]
            sparse_vector = self.create_sparse_vector(doc)

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
        self.qdrant_client.upsert(
            collection_name=Collection_Name,
            points=points
        )

        logging.info(f"Upserted {len(points)} points with dense and sparse vectors into Qdrant vector database.")

    def indexing(self) -> None:
        """
        Index the documents into the Qdrant vector database.
        """
        self.load_nodes()
        self.create_collection()
        self.insert_documents()
        logging.info("Indexing of the nodes is complete")
    

    def done() -> None:
        """When you inserted the vetors into the Qdrant Cluster, call this tool."""
        logging.info("Indexing of the nodes is complete")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    @staticmethod
    def IndexingAgent(self, state: dict) -> OpenAIAgent:
        """
        Create an OpenAI agent with the necessary tools and system prompt.
        """

        tools = [
            FunctionTool.from_defaults(fn=self.indexing),
            FunctionTool.from_defaults(fn=self.done),
        ]

        system_prompt = (f"""
        You are a helpful assistant that is indexing documents for a retrieval-augmented generation (RAG) system.
        Your task is to index the documents into a Qdrant cluster.
        To do this, you need to know the embedding model to use.
        You can ask the user to supply this.
        If the user supplies the embedding model, Use that embedding model and pass it to the "embedding_model" variable.
        Then, call the tool "indexing" to index the documents into the Qdrant cluster.
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
    agent = QdrantIndexing.IndexingAgent(state = state)
    response = agent.chat("I want to index the documents using the sentence-transformers/all-MiniLM-L6-v2 embedding model.")