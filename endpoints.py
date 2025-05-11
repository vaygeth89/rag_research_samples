import logging
from typing import Annotated, Union
from fastapi import FastAPI, File, UploadFile
from fastapi import FastAPI

from embedding import start_vector_store
from file_management import save_file
from llm import generate_response
from models.conversation_message import ConversationMessage

app = FastAPI()

@app.post("/start_embedding")
def start_embedding(query:str):
    vector_store =start_vector_store("./documents/sample.pdf")
    answer = generate_response(vector_store, query)
    logging.info(f"Answer: {answer}")
    # conversation_message = ConversationMessage(
    #   content=str(answer["output_text"]),
    # )
    return answer
    # return ConversationMessage(content=str(answer["output_text"]))


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    await save_file(file)
    return {"filename": file.filename}


@app.post("/embedding_document/")
async def create_upload_file(file: UploadFile, query: str):
    file_path =await save_file(file)
    vector_store =start_vector_store(file_path)
    answer = generate_response(vector_store, query)
    logging.info(f"Answer: {answer}")
    # conversation_message = ConversationMessage(
    #   content=str(answer["output_text"]),
    # )
    return ConversationMessage(content=str(answer["output_text"]))
