from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.query_engine import CustomQueryEngine
from retriever_agent import RetrieverAgent
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import BaseSynthesizer
import os


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

def prompt_generation(search_type, query, reranking_model):
    """
    Generate the prompt for the given search type, query, and reranking model.
    """
    state = {}
    retriever_agent = Retriever(state)
    reranked_documents = retriever_agent.retriever(search_type, query, reranking_model)

    context = "\n\n".join(reranked_documents)

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

def GenerationAgent(state: dict) -> OpenAIAgent:
    """
    Define the GenerationAgent for generating explanations based on the user's query, search type, and reranking model.
    """
    def generation(search_type: str, query: str, reranking_model: str):
        """
        Generate an explanation based on the given search type, query, and reranking model.
        """
        prompt = prompt_generation(search_type, query, reranking_model)
        print("Starting the search and retrieval process")
        response = create_query_engine(prompt)
        print("Retrieved the respopnse from LLMs")
        return print(response)

    def done() -> None:
        """
        Signal that the retrieval process is complete and update the state.
        """
        logging.info("Retrieval process is complete and updating the state")
        state["current_speaker"] = None
        state["just_finished"] = True

    tools = [
        FunctionTool.from_defaults(fn=generation),
        FunctionTool.from_defaults(fn=done),
    ]

    system_prompt = f"""
    You are a helpful assistant that is performing search and retrieval tasks for a retrieval-augmented generation (RAG) system.
    Your task is to retrieve documents based on the user's query, search type, and reranking model.
    To do this, you need to know the search type, query, and reranking model.
    You can ask the user to supply these details.
    If the user supplies the necessary information, then call the tool "generation" using the provided details to perform the search and retrieval process.
    The current user state is:
    {pprint.pformat(state, indent=4)}
    When you have completed the retrieval process, call the tool "done" to signal that you are done.
    If the user asks to do anything other than retrieve documents, call the tool "done" to signal that some other agent should help.
    """

    return OpenAIAgent.from_tools(
        tools,
        llm=OpenAI(model="gpt-3.5-turbo"),
        system_prompt=system_prompt,
    )

if __name__ == '__main__':
    state = {}
    agent = GenerationAgent(state=state)
    response = agent.chat("I want to query what is self-RAG? with Hybrid search and following with CrossEncoder Reranking model")
    
