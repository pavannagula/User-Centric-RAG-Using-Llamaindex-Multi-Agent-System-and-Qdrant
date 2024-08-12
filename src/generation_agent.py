from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.query_engine import CustomQueryEngine
from retriever_agent import Retriever
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.tools import FunctionTool
import os
import pprint
import logging

# Set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


load_dotenv()

def prompt_template():
    """
    Define the prompt template for generating explanations based on the context and query.
    """
    prompt_str = """
    You are an AI assistant specializing in explaining complex topics related to Retrieval-Augmented Generation(RAG).
    Your task is to provide a clear, concise, and informative explanation based on the following context and query.

    Context:
    {context_str}

    Query: {query_str}

    Please follow these guidelines in your response:
    1. Start with a brief overview of the concept mentioned in the query.
    2. Provide at least one concrete example or use case to illustrate the concept.
    3. If there are any limitations or challenges associated with this concept, briefly mention them.
    4. Conclude with a sentence about the potential future impact or applications of this concept.

    Your explanation should be informative yet accessible, suitable for someone with a basic understanding of RAG.
    If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer,
    and only respond based on the given context.

    Response:
    """
    prompt_tmpl = PromptTemplate(prompt_str)
    return prompt_tmpl

def prompt_generation(state):
    """
    Generate the prompt for the given search type, query, and reranking model.
    """
    state = state
    retriever_agent = Retriever(state)
    reranked_documents = retriever_agent.retriever()

    context = "\n\n".join(reranked_documents)
    query = state.get('query')
    prompt_templ = prompt_template().format(context_str=context, query_str=query)

    return prompt_templ

class RAGStringQueryEngine(CustomQueryEngine):
    llm: OpenAI
    response_synthesizer: BaseSynthesizer

    def custom_query(self, prompt: str) -> str:
        """
        Generate a response for the given prompt using the LLM and response synthesizer.
        """
        response = self.llm.complete(prompt)
        summary = self.response_synthesizer.get_response(query_str=str(response), text_chunks=str(prompt))

        return str(summary)
    
def create_query_engine(prompt: str):
    """
    Create a query engine for generating responses based on the given prompt.
    """
    llm = OpenAI(model="gpt-3.5-turbo")
    response_synthesizer = TreeSummarize(llm=llm)

    query_engine = RAGStringQueryEngine(
        llm=llm,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(prompt)
    return response.response

def generation(state):
    """
    Generate an explanation based on the given search type, query, and reranking model.
    """
    prompt = prompt_generation(state)
    print("Passing the ReRanked documents to the LLM")
    response = create_query_engine(prompt)
    print("Retrieved the summarized response from LLMs")
    #logger.info("Response:")
    #logger.info(response)
    return response


def GenerationAgent(state: dict) -> OpenAIAgent:
    """
    Define the GenerationAgent for generating explanations based on the user's query, search type, and reranking model.
    """

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

    def generate_response(state):
        response = generation(state)
        print(state)
        print(f"Response is generated and Here is the answer to your query:{response}")

    def done(state):
        """
        Signal that the retrieval process is complete, update the state, and return the response to the user.
        """
        print("Retrieval and Generation process is complete and updating the state")
        state["current_speaker"] = None
        state["just_finished"] = True

    tools = [
        FunctionTool.from_defaults(fn=has_query),
        FunctionTool.from_defaults(fn=has_search_type),
        FunctionTool.from_defaults(fn=has_reranking_model),
        FunctionTool.from_defaults(fn=generate_response),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = f"""
    You are a helpful assistant that is performing retrieval and generation tasks for a retrieval-augmented generation (RAG) system.
    Your task is to retrieve documents based on the user's query, search type, and reranking model, and then generate a response based on the retrieved documents.
    To do this, you need to know the search type, query, and reranking model.
    * If they want to query the documents, but has_query, has_search_type, or has_reranking_model returns false. Then, You can ask the user to supply these details.    
    If the user supplies the necessary information, and make sure that has_query, has_search_type and has_reranking_model are not none,
    then call the tool "generate_response" using the provided details to perform the retrieval and generation process.
    The current user state is:
    {pprint.pformat(state, indent=4)}
    When you have completed the generation process, call the tool "done" to signal that you are done and return the response to the user.
    If the user asks to do anything other than retrieve documents, call the tool "done" with an empty string as an argument to signal that some other agent should help.
    """


    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        system_prompt=system_prompt,
    )

if __name__ == '__main__':
    state = {   'chunk_overlap': None,
    'chunk_size': None,
    'current_speaker': None,
    'embedding_model': None,
    'input_dir': None,
    'just_finished': False,
    'query': 'what is self-RAG?',
    'reranking_model': None,
    'search_type': None,
    }
    agent = GenerationAgent(state=state)
    response = agent.chat("I want to query what is a Ragnarök framework? Also can you use hybrid search along with crossencoder reranking model")
