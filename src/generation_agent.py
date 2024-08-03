from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core.query_engine import CustomQueryEngine
from rerank import reranking
from Retriever import Hybrid_search
from llama_index.llms.mistralai import MistralAI
from dotenv import load_dotenv
from llama_index.core.response_synthesizers import BaseSynthesizer
import os

from src.retriever_agent import query_hybrid_search, metadata_filter

load_dotenv()

def prompt_template_generation():
    def prompt_template() -> None:
       
        prompt_str = """You are an AI assistant specializing in explaining complex topics related to Retrieval-Augmented Generation(RAG). Your task is to provide a clear, concise, and informative explanation based on the following context and query.

        Context:
        {context_str}

        Query: {query_str}

        Please follow these guidelines in your response:
        1. Start with a brief overview of the concept mentioned in the query.
        2. Provide at least one concrete example or use case to illustrate the concept.
        3. If there are any limitations or challenges associated with this concept, briefly mention them.
        4. Conclude with a sentence about the potential future impact or applications of this concept.

        Your explanation should be informative yet accessible, suitable for someone with a basic understanding of RAG. If the query asks for information not present in the context, please state that you don't have enough information to provide a complete answer, and only respond based on the given context.

        Response:
        """
        prompt_tmpl = PromptTemplate(prompt_str)

    def prompt_generation(query: str, filename: str):
        metadata_filter = metadata_filter(filename)
        results = query_hybrid_search(query, metadata_filter)
        
        reranked_documents = reranker.rerank_documents(query, results)
        
        context = "/n/n".join(reranked_documents)

        prompt_templ = prompt_tmpl.format(context_str=context, query_str=query)

        return prompt_templ

class RAGStringQueryEngine(CustomQueryEngine):
    llm: MistralAI
    response_synthesizer: BaseSynthesizer

    def custom_query(prompt: str) -> str:
        response = llm.complete(prompt)
        summary = response_synthesizer.get_response(query_str = str(response), text_chunks = str(prompt))

        return str(summary)
    
def create_query_engine(prompt: str):
    llm = MistralAI(model="open-mixtral-8x7b", api_key=os.environ.get('MistralAI'))
    response_synthesizer= TreeSummarize(llm=llm)

    query_engine = RAGStringQueryEngine(            
        llm=llm,
        response_synthesizer=response_synthesizer,
    )
    response = query_engine.query(prompt)
    return response.response

if __name__ == '__main__':
    query_str = "What are the evaluation results of -RAG?"
    filename = "-RAG.pdf"
    prompt_gen = prompt_template_generation()
    prompt = prompt_gen.prompt_generation(query = query_str, filename = filename)
    response = create_query_engine(prompt)
    print(response)