import json
import logging
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain

from models.conversation_message import ConversationMessageResponse


def generate_response(vector_store, query) -> ConversationMessageResponse:
    llm = OllamaLLM(model="qwen2.5:latest")
    # llm = OllamaLLM(model="llama3.2")
    chain = load_qa_chain(llm, chain_type="stuff", verbose=False)
    matching_docs = vector_store.similarity_search(query)
    response = chain.invoke(
        {
            "input_documents": matching_docs,
            "question": query
        }
    )
    logging.debug(f"Response: {response}")
    print(response)
    references_documents: list[str] = [str(doc) for doc in response["input_documents"]]
    return ConversationMessageResponse(content=response["output_text"], references=references_documents,
                                       question=query)
    # return response
