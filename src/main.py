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
from indexing_agent import QdrantIndexingAgent
from generation_agent import GenerationAgent


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
        You are a helpful assistant that is helping a user navigate the process of querying and indexing their documents using this customizable RAG application.
        Your job is to ask the user questions to figure out what they want to do, and give them the available scenario's.
        That includes:
        * pre-processing the documents/files and converting them into nodes based on user preferred chunking strategies.
        * Indexing the nodes into Qdrant vector database using user preferred embedding models.
        * generating a response to the user query using user preferred search type and reranking model.

        The current state of the user is:
        {pprint.pformat(state, indent=4)}
    """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
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
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.4),
        system_prompt=system_prompt,
    )

def orchestration_agent_factory(state: dict) -> OpenAIAgent:
    
    def has_embedding_model(embedding_model: str) -> bool:
        """Useful for checking if the user has specified an embedding model."""
        print("Orchestrator checking if embedding model is specified")
        state['embedding_model'] = embedding_model
        return (state["embedding_model"] is not None)
    def has_reranking_model(reranking_model: str) -> bool:
        """Useful for checking if the user has specified a reranking model."""
        print("checking if reranking model is specified")
        state['reranking_model'] = reranking_model
        return (state["reranking_model"] is not None)

    def has_search_type(search_type: str) -> bool:
        """Useful for checking if the user has specified a search type."""
        print("checking if search type is specified")
        state['search_type'] = search_type
        return (state["search_type"] is not None)    

    def has_query(query: str) -> bool:
        """Useful for checking if the user has specified query."""
        print("checking if query is specified")
        state['query'] = query
        return (state["query"] is not None)
      

    tools = [
        FunctionTool.from_defaults(fn=has_embedding_model),
        FunctionTool.from_defaults(fn=has_query),
        FunctionTool.from_defaults(fn=has_search_type),
        FunctionTool.from_defaults(fn=has_reranking_model),
    ]

    system_prompt =  (f"""
        You are the orchestration agent.
        Your job is to decide which agent to run based on the current state of the user and what they've asked to do. Agents are identified by short strings.
        What you do is return the name of the agent to run next. You do not do anything else.

        The current state of the user is:
        {pprint.pformat(state, indent=4)}

        If a current_speaker is already selected in the state, simply output that value.

        If there is no current_speaker value, look at the chat history and the current state and you MUST return one of these strings identifying an agent to run:
        * "{Speaker.Data_pre_processing.value}" - if the user wants to pre-process the documents into nodes
            * If they want to pre-process the documents, but has_input_dir, has_chunk_size, or has_chunk_overlap returns false, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Indexing.value}" - if the user wants to index the nodes into qdrant vector database
            * If they want to index the nodes, but has_embedding_model returns false, return "{Speaker.Concierge.value}" instead
        * "{Speaker.Generation.value}" - if the user wants to query the documents (requires query, search type, and reranking model)          
        * "{Speaker.Concierge.value}" - if the user wants to do something else, or hasn't said what they want to do, or you can't figure out what they want to do. Choose this by default.

        Output one of these strings and ONLY these strings, without quotes.
        NEVER respond with anything other than one of the above strings. DO NOT be helpful or conversational.
        """)

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0.4),
        system_prompt=system_prompt,
    )

def get_initial_state() -> dict:
    return {
        "input_dir": None,
        "chunk_size": None,
        "chunk_overlap": None,
        "embedding_model": None,
        "reranking_model": None,
        "search_type": None,
        "query": None,
        "current_speaker": None,
        "just_finished": False,
        "response": None,
    }


def run() -> None:
    state = get_initial_state()

    root_memory = ChatMemoryBuffer.from_defaults(token_limit=80000)

    first_run = True
    is_retry = False
    should_continue = True

    while should_continue:
        if first_run:
            user_msg_str = "Hello there!"
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
                if user_msg_str.lower() == "exit":
                    print("Exiting the conversation...")
                    should_continue = False
            state["just_finished"] = False
        else:
            user_msg_str = input("> ").strip()
            if user_msg_str.lower() == "exit":
                print("Exiting the conversation...")
                should_continue = False
        current_history = root_memory.get()

        if (state["current_speaker"]):
            print(f"There's already a speaker: {state['current_speaker']}")
            next_speaker = state["current_speaker"]
        else:
            print("No current speaker, asking orchestration agent to decide")
            orchestration_response = orchestration_agent_factory(state).chat(user_msg_str, chat_history=current_history)
            next_speaker = str(orchestration_response).strip()

        print(f"Next speaker: {next_speaker}")

        if next_speaker == Speaker.Data_pre_processing:
            print("Data pre-processing agent selected")
            current_speaker = DocumentPreprocessingAgent(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.Indexing:
            print("indexing agent is selected")
            current_speaker = QdrantIndexingAgent(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.Generation:
            print("Generation agent is selected")
            current_speaker = GenerationAgent(state)
            state["current_speaker"] = next_speaker
        elif next_speaker == Speaker.Concierge:
            print("Concierge agent selected")
            current_speaker = concierge_agent_factory(state)
        else:
            print("Orchestration agent failed to return a valid speaker; ask it to try again")
            is_retry = True
            continue

        pretty_state = pprint.pformat(state, indent=4)
        
        print(f"State: {pretty_state}")

        
        response = current_speaker.chat(user_msg_str, chat_history=current_history)
        print(str(response))

        
        new_history = current_speaker.memory.get_all()
        root_memory.set(new_history)

if __name__ == '__main__':
    print("Testing the Application")
    run()