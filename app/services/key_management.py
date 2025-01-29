#from db.models import Key
from app.db.models import Key
#from core.config import settings
from app.core.config import settings
from keycove import encrypt, decrypt, hash, generate_token
from sqlalchemy.orm import Session

""""Revisar que pasa con los datos encriptados y hasheados"""

def create_api_key(db: Session):
    api_key = generate_token()
    hashed_key = hash(api_key)
    encrypted_key = encrypt(api_key, settings.SECRET_KEY)
    new_key = Key(hashed_key=hashed_key, encrypted_key=encrypted_key)

    db.add(new_key)
    db.commit()
    db.refresh(new_key)

    return {"api_key": api_key}

def decrypt_api_key(api_key: str, db: Session):
    print(hash(api_key))
    key = db.query(Key).filter(Key.hashed_key == hash(api_key)).first()
    return decrypt(key.encrypted_key, settings.SECRET_KEY)

def get_keys(db: Session):
    keys = db.query(Key).all()
    return {"keys": [key.hashed_key for key in keys]}
