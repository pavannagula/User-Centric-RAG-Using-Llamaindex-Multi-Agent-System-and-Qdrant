import streamlit as st
from main import orchestration_agent_factory, get_initial_state, Speaker
from main import concierge_agent_factory
from main import continuation_agent_factory
from colorama import Fore, Style

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI 
from llama_index.agent.openai import OpenAIAgent
from document_pre_processing_agent import DocumentPreprocessingAgent
from indexing_agent import QdrantIndexingAgent
from generation_agent import GenerationAgent

# Title
st.set_page_config(page_title="Customize RAG with Multi Agents using Llamaindex and Qdrant", layout="wide")
st.title("Chat with your Multi Agentic RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize state
if "state" not in st.session_state:
    st.session_state.state = get_initial_state()

# Initialize root memory
if "root_memory" not in st.session_state:
    st.session_state.root_memory = ChatMemoryBuffer.from_defaults(token_limit=125000)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_msg_str = st.chat_input("Hello there!")

if user_msg_str:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_msg_str})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_msg_str)

    # Get current history
    current_history = st.session_state.root_memory.get()

    # Who should speak next?
    if st.session_state.state["current_speaker"]:
        next_speaker = st.session_state.state["current_speaker"]
    else:
        orchestration_response = orchestration_agent_factory(st.session_state.state).chat(user_msg_str, chat_history=current_history)
        next_speaker = str(orchestration_response).strip()

    # Select the current speaker
    if next_speaker == Speaker.Data_pre_processing:
        current_speaker = DocumentPreprocessingAgent(st.session_state.state)
        st.session_state.state["current_speaker"] = next_speaker
    elif next_speaker == Speaker.Indexing:
        current_speaker = QdrantIndexingAgent(st.session_state.state)
        st.session_state.state["current_speaker"] = next_speaker
    elif next_speaker == Speaker.Generation:
        current_speaker = GenerationAgent(st.session_state.state)
        st.session_state.state["current_speaker"] = next_speaker
    elif next_speaker == Speaker.Concierge:
        current_speaker = concierge_agent_factory(st.session_state.state)
    else:
        st.write("Orchestration agent failed to return a valid speaker; ask it to try again")
        st.session_state.state["current_speaker"] = None
        st.session_state.state["just_finished"] = False
        st.rerun()

    # Chat with the current speaker
    response = current_speaker.chat(user_msg_str, chat_history=current_history)
    print(f"Response: {response}")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write(str(response))

    # Update chat history
    new_history = current_speaker.memory.get_all()
    st.session_state.root_memory.set(new_history)

    # Update state
    st.session_state.state = st.session_state.state

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
