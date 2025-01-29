from fastapi import FastAPI
#from api import endpoints
from app.api import endpoints
#from db import models
from app.db import models
#from db.session import engine
from app.db.session import engine
import uvicorn

#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

models.Base.metadata.create_all(bind=engine)

# Incluye rutas de la API
app.include_router(endpoints.router)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
