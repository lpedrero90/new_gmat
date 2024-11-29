from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# URL de conexión a PostgreSQL en local
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:gmat@localhost:5432/gmat"

# Crear el motor de base de datos con PostgreSQL
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Configurar la sesión para conectarse con PostgreSQL
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarar la base
Base = declarative_base()

# Definir la función get_db
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
