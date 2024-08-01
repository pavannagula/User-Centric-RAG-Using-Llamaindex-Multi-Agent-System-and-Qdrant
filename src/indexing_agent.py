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

def DocumentPreprocessingAgent(state: dict) -> OpenAIAgent:

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

            logging.info(f"Loaded {len(nodes)} the nodes from JSON file")

        except Exception as e:
            logging.error(f"Error loading nodes from JSON file: {e}")
            raise

        return documents, metadata

    def client_collection(embedding_model, documents, metadata):
        qdrant_client = QdrantClient(
            url=Qdrant_URL,
            api_key=Qdrant_API_KEY)

        embedding_model = TextEmbedding(model_name=embedding_model)
        sparse_embedding_model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
        qdrant_client.set_model(embedding_model)
        qdrant_client.set_sparse_model(sparse_embedding_model)

        try:
            qdrant_client.recreate_collection(
                collection_name="Hybrid_RAG_Collection",
                vectors_config=qdrant_client.get_fastembed_vector_params(),
                sparse_vectors_config=qdrant_client.get_fastembed_sparse_vector_params(),
            )

            ids = qdrant_client.add(
                collection_name="Hybrid_RAG_Collection",
                documents=documents,
                metadata=metadata,
                ids=tqdm(range(len(documents))),
            )

            logging.info(f"Inserted {len(ids)} vectors into Qdrant cluster")

        except Exception as e:
            logging.error(f"Error inserting vectors into Qdrant cluster: {e}")
            raise

    def indexing(embedding_model):
        documents, metadata = load_nodes()
        logging.info("Loaded the nodes from json file")
        client_collection(embedding_model, documents, metadata)
        logging.info("Inserted the documents into the Qdrant Cluster")

    def done() -> None:
        """When you inserted the vetors into the Qdrant Cluster, call this tool."""
        logging.info("Indexing of the nodes is complete")
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
    If the user supplies the embedding model, call the tool "indexing" with this parameter to index the documents into the Qdrant cluster.
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
