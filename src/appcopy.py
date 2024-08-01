"""
A multi-agent conversational system for navigating a complex task tree.
In this demo, the user is navigating to create a custom RAG application.
AS RAG Application consists of multiple componenets to build based on user preference. 
I'm going to ask the user following components to specify:
    - Data
    - chunking details
    - Embeddings
    - Reranking Models
    - LLMs

Note: You can modify the user components as per your requirements.

Agent 1: pre-process the documents into nodes.

Agent 2: Embedding and indexing the nodes into the qdrant vector database

Agent 3: Retriever which uses the embedding models and converts the query to embeddings and clubs them
metadata filters to search the qdrant vector database

Agent 4: Reranking model defining based on the user preferred reranking model

Agent 5: Generates the response to the user query using the pre defined prompt templates and 
feeds it to the LLM based on the user prereferred LLM model

Concierge agent: a catch-all agent that helps navigate between the other 4.

Orchestration agent: decides which agent to run based on the current state of the user.
"""

from dotenv import load_dotenv
load_dotenv()

from enum import Enum
from typing import List
import pprint
from colorama import Fore, Back, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent
from document_pre_processing_agent import DocumentPreprocessingAgent


class Speaker(str, Enum):
    Data_pre_processing = "data_pre_processing"
    Indexing = "indexing"
    Retriever = "retriever"
    ReRanking = "reranking"
    Generation = "generation"
    Concierge = "Concierge"
    ORCHESTRATOR = "orchestrator"


def concierge_agent_factory(state: dict) -> OpenAIAgent:
    def dummy_tool() -> bool:
        """A tool that does nothing."""
        print("Doing nothing.")

    tools = [
        FunctionTool.from_defaults(fn=dummy_tool)
    ]

    system_prompt = (f"""
        You are a helpful assistant that is helping a user navigate the process of creating a custom RAG application.
        Your job is to ask the user questions to figure out what they want to do, and give them the available options.
        That includes
        * pre-processing the documents into nodes
        * embedding and indexing the nodes into a vector database
        * retrieving relevant nodes based on a query
        * reranking the retrieved nodes based on relevance
        * generating a response to the user query using a language model

        The current state of the user is:
        {pprint.pformat(state, indent=4)}
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o"),
        system_prompt=system_prompt,
    )

def continuation_agent_factory(state: dict) -> OpenAIAgent:
    def dummy_tool() -> bool:
        """A tool that does nothing."""
        print("Doing nothing.")

    tools = [
        FunctionTool.from_defaults(fn=dummy_tool)
    ]

    system_prompt = (f"""
        The current state of the user is:
        {pprint.pformat(state, indent=4)}
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o", temperature=0.4),
        system_prompt=system_prompt,
    )

def orchestration_agent_factory(state: dict) -> OpenAIAgent:
    def has_input_dir() -> bool:
        """Useful for checking if the user has specified an input directory."""
        print("Orchestrator checking if input directory is specified")
        return (state["input_dir"] is not None)

    def has_chunk_size() -> bool:
        """Useful for checking if the user has specified a chunk size."""
        print("Orchestrator checking if chunk size is specified")
        return (state["chunk_size"] is not None)

    def has_chunk_overlap() -> bool:
        """Useful for checking if the user has specified a chunk overlap."""
        print("Orchestrator checking if chunk overlap is specified")
        return (state["chunk_overlap"] is not None)

    def has_embedding_model() -> bool:
        """Useful for checking if the user has specified an embedding model."""
        print("Orchestrator checking if embedding model is specified")
        return (state["embedding_model"] is not None)

    def has_reranking_model() -> bool:
        """Useful for checking if the user has specified a reranking model."""
        print("Orchestrator checking if reranking model is specified")
        return (state["reranking_model"] is not None)

    def has_llm_model() -> bool:
        """Useful for checking if the user has specified a language model."""
        print("Orchestrator checking if language model is specified")
        return (state["llm_model"] is not None)

    tools = [
        FunctionTool.from_defaults(fn=has_input_dir),
        FunctionTool.from_defaults(fn=has_chunk_size),
        FunctionTool.from_defaults(fn=has_chunk_overlap),
        FunctionTool.from_defaults(fn=has_embedding_model),
        FunctionTool.from_defaults(fn=has_reranking_model),
        FunctionTool.from_defaults(fn=has_llm_model),
    ]

    system_prompt = (f"""
        You are the orchestration agent.
        Your job is to decide which agent to run based on the current state of the user and what they've asked to do. Agents are identified by short strings.
        What you do is return the name of the agent to run next. You do not do anything else.

        The current state of the user is:
        {pprint.pformat(state, indent=4)}

        If a current_speaker is already selected in the state, simply output that value.

        If there is no current_speaker value, look at the chat history and the current state and you MUST return one of these strings identifying an agent to run:
        * "{Speaker.Data_pre_processing.value}" - if the user wants to pre-process the documents into nodes
            * If they want to pre-process the documents, but they haven't specified an input directory, return "{Speaker.Concierge.value}" instead
            * If they want to pre-process the documents, but they haven't specified a chunk size or chunk overlap, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Indexing.value}" - if the user wants to embed and index the nodes into a vector database
            * If they want to embed and index the nodes, but they haven't specified an embedding model, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Retriever.value}" - if the user wants to retrieve relevant nodes based on a query
            * If they want to retrieve relevant nodes, but they haven't specified an embedding model, return "{Speaker.Concierge.value}" instead
        * "{Speaker.ReRanking.value}" - if the user wants to rerank the retrieved nodes based on relevance
            * If they want to rerank the retrieved nodes, but they haven't specified a reranking model, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Generation.value}" - if the user wants to generate a response to the user query using a language model
            * If they want to generate a response, but they haven't specified a language model, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Concierge.value}" - if the user wants to do something else, or hasn't said what they want to do, or you can't figure out what they want to do. Choose this by default.

        Output one of these strings and ONLY these strings, without quotes.
        NEVER respond with anything other than one of the above seven strings. DO NOT be helpful or conversational.
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-4o", temperature=0.4),
        system_prompt=system_prompt,
    )

def get_initial_state() -> dict:
    return {
        "session_token": None,
        "input_dir": None,
        "chunk_size": None,
        "chunk_overlap": None,
        "embedding_model": None,
        "reranking_model": None,
        "llm_model": None,
        "current_speaker": None,
        "just_finished": False,
    }

def get_agent(agent_name, state):
    agents = {
        "Data_pre_processing": DocumentPreprocessingAgent,
        "Indexing": IndexingAgent,
        "Retriever": RetrieverAgent,
        "ReRanking": RerankingAgent,
        "Generation": GenerationAgent,
        "Concierge": ConciergeAgent,
        # Add other agents here
    }
    return agents.get(agent_name, None)(state)

def run() -> None:
    state = get_initial_state()

    root_memory = ChatMemoryBuffer.from_defaults(token_limit=8000)

    first_run = True
    is_retry = False

    while True:
        if first_run:
            # if this is the first run, start the conversation
            user_msg_str = "Hello"
            first_run = False
        elif is_retry == True:
            user_msg_str = "That's not right, try again. Pick one agent."
            is_retry = False
        elif state["just_finished"] == True:
            print("Asking the continuation agent to decide what to do next")
            user_msg_str = str(continuation_agent_factory(state).chat("""
                Look at the chat history to date and figure out what the user was originally trying to do.
                They might have had to do some sub-tasks to complete that task, but what we want is the original thing they started out trying to do.
                Formulate a sentence as if written by the user that asks to continue that task.
                If it seems like the user really completed their task, output "no_further_task" only.
            """, chat_history=current_history))
            print(f"Continuation agent said {user_msg_str}")
            if user_msg_str == "no_further_task":
                user_msg_str = input(">> ").strip()
            state["just_finished"] = False
        else:
            # any other time, get user input
            user_msg_str = input("> ").strip()

        current_history = root_memory.get()

        # who should speak next?
        if (state["current_speaker"]):
            print(f"There's already a speaker: {state['current_speaker']}")
            next_speaker = state["current_speaker"]
        else:
            print("No current speaker, asking orchestration agent to decide")
            orchestration_response = orchestration_agent_factory(state).chat(user_msg_str, chat_history=current_history)
            next_speaker = str(orchestration_response).strip()

        print(f"Next speaker: {next_speaker}")

        agent_class = get_agent(next_speaker, state)
        if agent_class:
            current_speaker = agent_class
            state["current_speaker"] = next_speaker
        else:
            print("Orchestration agent failed to return a valid speaker; ask it to try again")
            is_retry = True
            continue

        pretty_state = pprint.pformat(state, indent=4)
        
        print(f"State: {pretty_state}")

        # chat with the current speaker
        response = current_speaker.chat(user_msg_str, chat_history=current_history)
        print(Fore.MAGENTA + str(response) + Style.RESET_ALL)

        # update chat history
        new_history = current_speaker.memory.get_all()
        root_memory.set(new_history)



if __name__ == '__main__':
    print("Testing the Application")
    run()