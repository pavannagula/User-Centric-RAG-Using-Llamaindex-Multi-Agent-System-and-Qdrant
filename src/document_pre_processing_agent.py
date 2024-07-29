from dotenv import load_dotenv
load_dotenv()
from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent

import os
import json
import re
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

def DocumentPreprocessingAgent(state: dict) -> OpenAIAgent:

    def documents_transformation(input_dir: str):
        print(f"Input directory: {input_dir}")
        documents = SimpleDirectoryReader(input_dir=input_dir).load_data()
        print(f"Loaded {len(documents)} documents")
        transformed_documents = []
        for doc in documents:
            transformed_content = doc.get_content().lower()
            transformed_content = re.sub(r'\s+', ' ', transformed_content)
            transformed_content = re.sub(r'[^\w\s]', '', transformed_content)
            transformed_documents.append(Document(text=transformed_content, metadata=doc.metadata))
        print(f"Transformed {len(documents)} documents")
        return transformed_documents

    def split_documents_into_nodes(documents, chunk_size, chunk_overlap):
        try:
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents(documents)
            return nodes
        except Exception as e:
            print(f"Error splitting documents into nodes: {e}")
            return []

    def save_nodes(nodes):
        try:
            output_file = r"C:\Users\pavan\Desktop\Generative AI\RAG-Automation-Using-Llamaindex-Agents-and-Qdrant\data\nodes.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            nodes_dict = [node.dict() for node in nodes]
            with open(output_file, 'w') as file:
                json.dump(nodes_dict, file, indent=4)
            print(f"Saved nodes to {output_file}")
        except Exception as e:
            print(f"Error saving nodes to file: {e}")
    
    def process_documents(input_dir: str, chunk_size: int, chunk_overlap: int) -> None:
        input_dir = rf"{input_dir}"
        documents = documents_transformation(input_dir)
        print("Document Transformation is done")
        nodes = split_documents_into_nodes(documents, chunk_size, chunk_overlap)
        print("Transformed into nodes")
        save_nodes(nodes)
        print("saved the nodes")

    def done() -> None:
        """When you saved node to the output file, call this tool."""
        print("Document preprocessing is complete")
        state["current_speaker"] = None
        state["just_finished"] = True
    
    tools = [
        FunctionTool.from_defaults(fn=process_documents),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = (f"""
    You are a helpful assistant that is preprocessing documents for a retrieval-augmented generation (RAG) system.
    Your task is to preprocess the documents, split them into nodes, and save the nodes to a file.
    To do this, you need to know the path to the directory containing the PDF files, the chunk size, and the chunk overlap.
    You can ask the user to supply these.
    If the user supplies the input_dir, chunk_size, and chunk_overlap, call the tool "process_documents" with these parameters to perform transformation of documents, split them into nodes, and save the nodes to a file.
    The current user state is:
    {pprint.pformat(state, indent=4)}
    When you have transformed the documents, split them into nodes, and saved the nodes to a file, call the tool "done" to signal that you are done.
    If the user asks to do anything other than preprocess the documents, call the tool "done" to signal some other agent should help.
    """)


    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        system_prompt=system_prompt,
    )

'''
if __name__ == '__main__':
    state = {}
    agent = DocumentPreprocessingAgent(state)
    response = agent.chat("I want to preprocess the documents in C:\\Users\\pavan\\Desktop\\Generative AI\\RAG-Automation-Using-Llamaindex-Agents-and-Qdrant\\data with a chunk size of 800 and a chunk overlap of 50.")
'''