import json
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from utils.chat_engine import handle_session
from utils.utils import gcs_fs, session_id_wrapper_json

logging.getLogger().setLevel(logging.INFO)

app = FastAPI()

origins = [
    "https://aisha-pd.technology-robot.com",
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/message")
async def message(
    question: str,
    session_id: str,
):
    store_session_path = os.path.join(os.environ["chats_path"], session_id_wrapper_json(session_id))
    answer = handle_session(question, store_session_path)
    return answer

@app.get("/chat_history")
async def chat_history(
    session_id: str,
):
    store_session_path = os.path.join(os.environ["chats_path"], session_id_wrapper_json(session_id))
    if gcs_fs.exists(store_session_path):
        with gcs_fs.open(store_session_path, 'r') as f_p:
            chat_history = json.load(f_p)
        return chat_history
    return []

@app.get("/delete_chat_history")
async def delete_chat_history(
    session_id: str,
):
    store_session_path = os.path.join(os.environ["chats_path"], session_id_wrapper_json(session_id))
    if gcs_fs.exists(store_session_path):
        gcs_fs.rm(store_session_path)
        return True
    return False
