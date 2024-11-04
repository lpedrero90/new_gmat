from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session
from db.models import Key, Permission
from db.session import get_db
from services.key_management import hash

def verify_api_key(api_key: str = Header(None), db: Session = Depends(get_db)) -> None:
    key = db.query(Key).filter(Key.hashed_key == hash(api_key)).first()
    if not key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Invalid API key",
        )
    
    # Verificar permisos y obtener el user_id relacionado
    permission = db.query(Permission).filter(Permission.api_key_id == key.id).first()
    if not permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No permission associated with this API key."
        )
    
    # Retornar el user_id asociado con la API key
    return permission.user_id
