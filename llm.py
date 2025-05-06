import logging
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain


def generate_response(vector_store, query):
    # llm = OllamaLLM(model="qwen2.5:latest")
    llm = OllamaLLM(model="llama3.2")
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
    matching_docs = vector_store.similarity_search(query)
    response = chain.invoke(
        {
        "input_documents": matching_docs, 
        "question": query
        }
    )
    return response