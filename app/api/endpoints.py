from fastapi import APIRouter, Depends, File, UploadFile
from typing import List
from app.core.security import verify_api_key
from app.services import ai, key_management
from app.db.session import get_db
from sqlalchemy.orm import Session
from app.services.key_management import create_api_key, decrypt_api_key, get_keys
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str
    user_gmat: int

router = APIRouter()

@router.post("/upload_file")
async def upload_file(
    files: List[UploadFile] = File(...), 
    user_id: int = Depends(verify_api_key),
    db: Session = Depends(get_db)
):

    return await ai.process_upload(files, user_id, db)

@router.post("/chat-doc")
async def chat_doc_endpoint(
    request: ChatRequest, 
    user_id: int = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    return await ai.get_bot_response(request.question, user_id, db)

@router.post("/chat-sql")
async def chat_sql_endpoint(
    request: ChatRequest, 
    user_id: int = Depends(verify_api_key),
    db: Session = Depends(get_db)
):
    return await ai.get_sql_bot_response(request.question, request.user_gmat, db)



@router.post("/create_api_key")
def create_api_key_endpoint(db: Session = Depends(get_db)):
    return create_api_key(db)

@router.get("/decrypt_api_key")
def decrypt_api_key_endpoint(
    api_key: str, 
    verify_api_key: str = Depends(verify_api_key),
    db: Session = Depends(get_db),
):
    return decrypt_api_key(api_key, db)

@router.get("/keys")
def get_keys_endpoint(db: Session = Depends(get_db)):
    """
    This function retrieves all the hashed keys from the database.

    Parameters:
    db (Session): The database session to use for querying the database.

    Returns:
    dict: A dictionary with all the hashed keys.
    """
    return get_keys(db)
    


