
import logging
import os

from fastapi import UploadFile


async def save_file(file:UploadFile):
    default_files_path ="./documents/"
    if not os.path.exists(default_files_path):
        os.makedirs(default_files_path)
    file_path = os.path.join(default_files_path, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    logging.info(f"File saved to {file_path}")
    return file_path