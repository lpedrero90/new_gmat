from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, chain
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.chains import TransformChain
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import BaseModel, Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)


from keycove import decrypt, hash, generate_token
from typing import List, Optional
from uuid import uuid4
from fastapi import UploadFile
from openai import OpenAI

from sqlalchemy.orm import Session
#from db.models import Topic, Document, User
from app.db.models import Topic, Document, User

import base64
import os
import requests
import json
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='ai.log', level=logging.DEBUG) #encoding='utf-8', 



# Set up OpenAI API configuration (you may choose to load these from environment variables)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_CREDENTIAL = os.getenv("EMBEDDING_CREDENTIAL")
PINECONE_CREDENTIAL = os.getenv("PINECONE_CREDENTIAL")
DB_CREDENTIAL = os.getenv("DB_CREDENTIAL")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
COLLECTION_NAME = "example_collection"

# Detalles de conexión
#server = 'win201.loading.es'
server = 'sgcconsulting.loading.net'
port = 1433
database = 'Gmat1'
username = 'otrodbo'
password = 'Laura2024.'

# Crear la conexión con SQLAlchemy
db_uri = (
    f"mssql+pyodbc://{username}:{password}@{server}:{port}/{database}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)
# Añadir tabla de pesos
db_sql = SQLDatabase.from_uri(
    db_uri,
    include_tables=['v_PesajeLaura', 'v_EPISLaura'], #Añadir la de EPIs
    view_support = True
)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db"
)
client = OpenAI(api_key=OPENAI_API_KEY)


class TextExtract(BaseModel):
    title: str = Field(description="The perceived title on the image")
    main_text: str = Field(description="The main text on the file")
    main_text_en: str = Field(description="The main text on the file translated to English")
    objects_in_image: str = Field(description="Any other objects observed in the image")


class TicketInformation(BaseModel):
    empresa: str = Field()
    direccion: Optional[str] = Field()
    cif: Optional[str] = Field()
    fecha: str = Field()
    base_imponible: str = Field()
    iva_porcentaje: str = Field()
    iva_cantidad: str = Field()
    descuento: Optional[str] = Field()
    total: str = Field()

class Item(BaseModel):
    dayWeek: str = Field(description="Día de la semana de lunes a viernes.")
    time: str = Field(description="tiempo que tardan en hacer la tarea, con el formato, por ejemplo 15min, 30min, 1h...")
    code: str = Field(description="codigo de la tarea, por ejemplo: I081911, I082144M, I082050M")
    quantity: int = Field(description="cantidad de unidades realizadas en numero entero")

class ReportInformation(BaseModel):
    section: str = Field(description="seccion en la que se realiza la tarea, por ejemplo MONTAJE")
    worker: str = Field(description="Nombre del trabajador, por ejemplo NATALIA")
    date: str = Field(description="fecha, con el formato dd/mm/yy, por ejemplo 13/06/25")
    role: str = Field(description="cargo del trabajador, por ejemplo OPERARIO")
    items: list[Item] = Field(description="lista de tareas realizadas en el turno")


class ImageInformation(BaseModel):
    numero_informe: str = Field()
    empresa_distribucion_electrica: str = Field()
    nombre_contratista: str = Field()
    situacion: str = Field()
    supervisado_por:  str = Field()
    fecha:  str = Field()
    linea:  str = Field()
    desde:  str = Field()
    desde_conexion:  str = Field()
    desde_celda:  str = Field()
    desde_senalizacion:  str = Field()
    hasta:  str = Field()
    hasta_conexion:  str = Field()
    hasta_celda:  str = Field()
    hasta_senalizacion:  str = Field()
    fabricante:  str = Field()
    aislamiento:  str = Field()
    tension_nominal:  str = Field()
    conductor:  str = Field()
    pantalla:  str = Field()
    tension_servicio:  str = Field()
    año_fabricacion:  str = Field()
    longitud:  str = Field()
    colores: str = Field()
    colores_fase_1:  str = Field()
    colores_fase_2:  str = Field()
    colores_fase_3:  str = Field()
    continuidad: str = Field()
    continuidad_malla_fase_1: str = Field()
    continuidad_malla_fase_2: str = Field()
    continuidad_malla_fase_3: str = Field()
    resistencia: str = Field() 
    resistencia_fase_1: str = Field()
    resistencia_fase_2: str = Field()
    resistencia_fase_3: str = Field()
    rigidez_dielectrica: str = Field()
    rigidez_dielectrica_fase_1: str = Field()
    rigidez_dielectrica_fase_2: str = Field()
    rigidez_dielectrica_fase_3: str = Field()
    rigidez_tiempo: str = Field()
    rigidez_tension: str = Field()
    ensayo_l1: str = Field()
    ensayo_l2: str = Field()
    ensayo_l3: str = Field()
    ensayo_tecnologia: str = Field()
    ensayo_modo: str = Field()
    ensayo_tension: str = Field()
    observaciones: str = Field()

parser = JsonOutputParser(pydantic_object=ImageInformation)
ticket_parser = JsonOutputParser(pydantic_object=TicketInformation)
report_parser = JsonOutputParser(pydantic_object=ReportInformation)

async def extract_main_topic(text: str, db: Session) -> str:
    """
    Extracts the main topic from the text using the chat model.

    Parameters:
    text (str): Text to analyze
    client: OpenAI client object

    Returns:
    str: Main topic extracted from the text
    """
    logger.info('Extract topic')

    response = client.chat.completions.create(model=CHAT_MODEL,
                                              messages=[
                                                  {"role": "user", "content": f"What is the main topic of the following text? {text}. Write only the topic in Spanish and be specific."}
                                            ])
    try:                                          
        main_topic = response.choices[0].message.content
    except Exception as e:
        logger.error('Error in response from GPT: {e}')

    logger.info(f'Topic already extracted: {main_topic}')
    # Verificar si el tópico ya existe en la base de datos
    existing_topic = db.query(Topic).filter(Topic.name == main_topic).first()
    if not existing_topic:
        # Si el tópico no existe, chequear si existe algún topic similar, sino se crea de nuevo
        # Chequear si existe algún topic válido (PENDING)

        new_topic = Topic(name=main_topic)
        db.add(new_topic)
        try:
            db.commit()
        except Exception as e:
            logger.error('Error commiting topic into db: {e}')
        
        db.refresh(new_topic)  # Actualiza el objeto con la información de la BD
        logger.info('Topic saved')
    
    return main_topic 


async def load_pdf_file(file: UploadFile, user_id: int, db: Session) -> str:
    # Aquí puedes agregar la lógica para guardar el archivo en el servidor
    # Por ejemplo, guardar el archivo en un directorio específico
    logger.info('Load PDF files')
    with open(f"app/uploads/{file.filename}", "wb") as buffer:
        try:                                          
            buffer.write(await file.read())
        except Exception as e:
            logger.error('Error writing in buffer: {e}')
        
    file_path = f"app/uploads/{file.filename}"

    logger.info('PyPDFloader')
    loader = PyPDFLoader(file_path)
    try:                                          
        pages = loader.load_and_split()
    except Exception as e:
        logger.error('Error loading snd splitting pages: {e}')
    

    # Extract the main topic from the content of the first page
    logger.info('Extract main topic')
    main_topic = await extract_main_topic(pages[0].page_content, db)

    logger.info('Loop over pages')
    for page in pages:
        page.metadata['topic'] = main_topic

    logger.info('Get topics')
    topic = db.query(Topic).filter(Topic.name == main_topic).first()
    if not topic:
        logger.error('Error retrieving topic')
        raise ValueError("Error retrieving topic.")

    logger.info('Create new document')
    new_document = Document(filename=file.filename, user_id=user_id, topic_id=topic.id)
    db.add(new_document)
    try:
        db.commit()
    except Exception as e:
        logger.error('Error commiting document into db: {e}')

    return pages


# async def process_upload(files: List, user_id: int, db: Session) -> str:
#     """
#     Processes uploaded files, splits them, and saves to the vector store with extracted topics.

#     Parameters:
#     files (List): List of UploadFile objects
#     extractor_function: Function to extract main topic

#     Returns:
#     dict: Success message upon completion
#     """
#     logger.info('Process upload')
#     content = []
#     for file in files:
#         if file.filename.endswith('.txt'):
#             #content = await extractor_function(file)
#             pass
#         elif file.filename.endswith('.pdf'):
#             pages_from_pdf = await load_pdf_file(file, user_id, db)
#             content.extend(pages_from_pdf)  # Assuming load_pdf_file returns a list of pages
#         elif file.filename.endswith('.docx'):
#             #content = extractor_function(file)
#             pass
#         else:
#             logger.error('Unsopported file format')
#             raise ValueError("Unsupported file format.")

#     if not content:
#         logger.error('No pages to process')
#         raise ValueError("No pages to process.")

#     #uuids = [str(uuid4()) for _ in range(len(content))]
#     logger.info('Character Text splitter')
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
#     logger.info('Text splitter')
#     all_splits = text_splitter.split_documents(content)

#     logger.info('Match uuids')
#     uuids = [str(uuid4()) for _ in range(len(all_splits))]  # Match uuids to the number of splits
#     logger.info('Add documents to vector store')
#     vector_store.add_documents(documents=all_splits, ids=uuids)

#     return {"message": "Files uploaded and processed successfully."}


async def process_upload(files: List, user_id: int, db: Session) -> str:
    """
    Processes uploaded files, splits them, and saves to the vector store with extracted topics.

    Parameters:
    files (List): List of UploadFile objects
    extractor_function: Function to extract main topic

    Returns:
    dict: Success message upon completion
    """
    logger.info('Process upload')
    error = 0
    user = db.query(User).filter(User.id == user_id).first()
    DOC_STORE_ID = user.whabble_document_store_id
    API_URL = f"https://whabble.nevrom.com/api/v1/document-store/upsert/{DOC_STORE_ID}"
    API_KEY = user.whabble_apikey
    
    for file in files:
        if file.filename.endswith('.pdf'):
            with open(f"app/uploads/{file.filename}", "wb") as buffer:
                try:                                          
                    buffer.write(await file.read())
                except Exception as e:
                    logger.error('Error writing in buffer: {e}')

            #Vincular el archivo a lo de GMAT
            new_document = Document(filename=file.filename, user_id=user_id)
            db.add(new_document)
            try:
                db.commit()
            except Exception as e:
                logger.error('Error commiting document into db: {e}')
                
            file_path = f"app/uploads/{file.filename}"
    
            print(f"Procesando: {file.filename}")

            form_data = {
                "files": (f"{file.filename}", open(f"{file_path}", 'rb'))
            }

            loader = {
                "name": "pdfFile",
                "config": {
                } # you can leave empty to use default config
            }

            splitter = {
                "name": "recursiveCharacterTextSplitter",
                "config": {
                    "chunkSize": 1500,
                    "chunkOverlap": 750
                }
            }

            embedding = {
                "name": "openAIEmbeddings",
                "config": {
                    "modelName": "text-embedding-3-small",
                    "credential": EMBEDDING_CREDENTIAL
                }
            }

            vectorStore = {
                "name": "pinecone",
                "config": {
                    "pineconeIndex": "gmatwhabble",
                    "pineconeNamespace": "exver",
                    "searchType":"similarity",
                    "credential":  PINECONE_CREDENTIAL
                }
            }

            recordManager = {
                "name": "postgresRecordManager",
                "config": {
                    "cleanup": "none",
                    "credential": DB_CREDENTIAL,
                    "database": "retrieval",
                    "host":"localhost",
                    "namespace":"exver",
                    "port":"5432",
                    "sourceIdKey":"source",
                    "ssl":True
                }
            }

            body_data = {
                #"docId": DOC_LOADER_ID,
                #"replaceExisting": False,
                "loader": json.dumps(loader),
                "splitter": json.dumps(splitter),
                "embedding": json.dumps(embedding),
                "vectorStore": json.dumps(vectorStore),
                "recordManager": json.dumps(recordManager)
            }

            headers = {
                "Authorization": f"Bearer {API_KEY}"
            }

            try:  
                response = requests.post(
                    API_URL, 
                    files=form_data, 
                    data=body_data, 
                    headers=headers, 
                    verify=False
                )
               
                print(response)
            except Exception as e:
                error = error + 1
                logger.error(f'Error uploading file {file.filename}: {e}')
            


    return {"message": f"Files uploaded with {error} errors."}

async def delete_upload(file: int, user_id, db: Session):
    pass

async def get_bot_response(question: str, user_id: int, db: Session, history: list = []) -> str:
    """
    Generates a response for a given question based on retrieved documents.

    Parameters:
    question (str): The question asked by the user

    Returns:
    str: Response generated by the bot
    """
    logger.info('Get Bot response')
    user = db.query(User).filter(User.id == user_id).first()
    apikey = user.whabble_apikey
    chatflow = user.whabble_chatflow_id
    
    api_url = f"https://whabble.nevrom.com/api/v1/prediction/{chatflow}"
    headers = {
        "Authorization": f"{apikey}",
        "Content-Type": "application/json"
        }
    print(history)
    
    converted_history = []
    if history:
        for index, message in enumerate(history):
            role = "userMessage" if index % 2 == 1 else "apiMessage"
            converted_history.append({
                "role": role,
                "content": message
            })

    payload = json.dumps({
        "question": question,
        "history": converted_history
        })

    try:
        response = requests.request("POST", api_url, headers=headers, data=payload)
        res = response.json()
        print(res)
        return res['text']
    except Exception as e:
        print(e)
        return 'Ha habido un problema con los permisos del usuario'
    # Extraer los topics del usuario
    # logger.info('Get all user topics')
    # user_topics = db.query(Topic).filter(Topic.documents.any(Document.user_id == user_id)).all()  # Obtener topics del usuario
    # topics = [topic.name for topic in user_topics]  # Obtener nombres de los topics
    
    # #-------------------------
    # #Extract topics con GPT (ver si puedo hacer para extraer en base al contexto)
    # logger.info('Get topics similar to question')
    # topic_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    # similar_topic_doc = topic_retriever.invoke(question)  # Buscar el topic más similar a la pregunta
    # #-------------------------

    # # Obtener el topic más relevante basado en la búsqueda
    # logger.info('Get most relevant topic')
    # selected_topic = similar_topic_doc[0].metadata["topic"] if similar_topic_doc else None
 
    # # Extraer únicamente los documentos con el meta del topic
    # logger.info('Get more relevant documents within the topic')
    # retriever = vector_store.as_retriever(
    #     search_type="similarity", 
    #     search_kwargs={
    #         "k": 6,
    #         "filter": {"topic": selected_topic}
    #     }
    # )
    # retrieved_docs = retriever.invoke(question)
    

    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)

    # llm = ChatOpenAI(model=CHAT_MODEL)
    # system_prompt = (
    #     "You are an assistant for question-answering tasks. "
    #     "Use the following pieces of retrieved context to answer "
    #     "the question. If you don't know the answer, say that you "
    #     "don't know. Use three sentences maximum and keep the "
    #     "answer concise. Answer in Spanish."
    #     "\n\n"
    #     "{context}"
    # )

    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("system", system_prompt),
    #         ("human", "{question}"),
    #     ]
    # )

    # rag_chain = (
    #     {"context": retriever | format_docs, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # logger.info('Chain invoke')
    # response = rag_chain.invoke(question)
    return response

async def get_sql_bot_response(question: str, user_gmat: int, db: Session) -> str:
    logger.info('Get SQL Bot response')
    print("USER GMAT", user_gmat)

    question = question + f". Filtra en los datos where IdEmpresa={user_gmat}. No devuelvas en ningún caso el IdEmpresa en la respuesta."

    examples = [
        {
            "input": f"¿Cuántos cascos hay disponibles?",
            "query": f"""SELECT COUNT(*) AS TotalCascosDisponibles FROM v_EPISLaura 
                        WHERE IdEmpresa = 1 
                        AND (Tipo LIKE '%casco%' OR Tipo LIKE '%cascos%') 
                        AND Disponible = 'Disponible';"""
        },
        {
            "input": "¿Quién tiene cascos?",
            "query": """
                    SELECT QuienTiene, Tipo, Disponible 
                    FROM v_EPISLaura 
                    WHERE IdEmpresa = 371 
                    AND (Tipo LIKE '%casco%' OR Tipo LIKE '%cascos%') 
                    AND Disponible = 'No disponible' 
                    ORDER BY QuienTiene;
                    """
        },
        {
            "input": "¿Cuántos equipos hay retirados?",
            "query": """ SELECT count(idEquipo) FROM v_EPISLaura
                        WHERE idEmpresa=371 AND retirado=1
                    """
        },
        {
            "input": "¿Qué personas han tenido el equipo con número de serie 12326/0273?",
            "query": """ SELECT serie, tipo, fecha, quien FROM v_TrazabilidadLaura
                        WHERE idEmpresa=371 AND serie='12326/0273' order by fecha;
                    """
        },
        {
            "input": "¿Cuál es el próximo equipo que tiene que verificar Raúl Martín?",
            "query": """ SELECT serie, tipo, marca, modelo, proxima FROM v_EPISLaura
                        WHERE idEmpresa=371 AND quienTiene LIKE '%Raul martin%' AND proxima >= getDate();
                    """
        },
    ]

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.

    DO NOT answer with the IdEmpresa.

    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    
    # Run agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent_executor = create_sql_agent(llm, db=db_sql, prompt=full_prompt, agent_type="openai-tools", verbose=True, top_k=1000)
    response = agent_executor.invoke({
        "agent_scratchpad": "",  # Assuming this needs to be an empty string if not used
        "input": question  # Changed from "query" to "input"
    })
    logger.info('SQL chain invoke')

    return response['output']

def load_image(inputs: dict) -> dict:
    logger.info('Load image')
    image_path = inputs["file_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_base64 = encode_image(image_path)
    return {"image": image_base64}

@chain
def image_model(inputs: dict):
    logger.info('Image model')
    model = ChatOpenAI(
        temperature=1,
        model="gpt-4o"
    )
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": parser.get_format_instructions()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs['image']}"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content

async def get_data_json_response(file: UploadFile, category: str, user_gmat: int, db:Session) -> dict:
    logger.info('Get data JSON response')
    # Aquí puedes agregar la lógica para guardar el archivo en el servidor
    # Por ejemplo, guardar el archivo en un directorio específico
    with open(f"app/uploads/extract/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    file_path = f"app/uploads/extract/{file.filename}"

    #If category is type1
    logger.info('Parser')

    logger.info('Transformation Chain')
    load_image_chain = TransformChain(
        input_variables=["file_path"], output_variables=["image"], transform=load_image
    )

    vision_prompt = """Extrae los datos escritos a mano de la fotografía. Pon el valor null en caso de que el campo esté vacío, no ivnventes los datos."""
    vision_chain = load_image_chain | image_model | parser
    #try:
    logger.info('Vision chain')
    return vision_chain.invoke(
        {"file_path": f"{file_path}", "prompt": vision_prompt}
    )

@chain
def ticket_model(inputs: dict):
    logger.info('Ticket model')
    model = ChatOpenAI(
        temperature=1,
        model="gpt-4o-mini"
    )
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": ticket_parser.get_format_instructions()},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs['image']}"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content

async def get_ticket_json_response(file: UploadFile, user_gmat: int, db:Session) -> dict:
    logger.info('Get ticket JSON response')
    # Aquí puedes agregar la lógica para guardar el archivo en el servidor
    # Por ejemplo, guardar el archivo en un directorio específico
    with open(f"app/uploads/extract/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    file_path = f"app/uploads/extract/{file.filename}"

    #If category is type1
    logger.info('Parser')

    logger.info('Transformation Chain')
    load_image_chain = TransformChain(
        input_variables=["file_path"], output_variables=["image"], transform=load_image
    )

    vision_prompt = """Extrae los datos escritos a mano de la fotografía. Pon el valor null en caso de que el campo esté vacío, no ivnventes los datos."""
    vision_chain = load_image_chain | ticket_model | ticket_parser
    #try:
    logger.info('Vision chain')
    return vision_chain.invoke(
        {"file_path": f"{file_path}", "prompt": vision_prompt}
    )

@chain
def report_model(inputs: dict):
    logger.info('Report model')
    model = ChatOpenAI(
        temperature=1,
        model="gpt-4o-mini"
    )
    msg = model.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": inputs["prompt"]},
                    {"type": "text", "text": report_parser.get_format_instructions()},
                    {
                        "type": "image_url",    
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{inputs['image']}"
                        },
                    },
                ]
            )
        ]
    )
    return msg.content

async def get_report_json_response(file: UploadFile, user_gmat: int, db:Session) -> dict:
    logger.info('Get report JSON response')
    # Aquí puedes agregar la lógica para guardar el archivo en el servidor
    # Por ejemplo, guardar el archivo en un directorio específico
    with open(f"app/uploads/extract/{file.filename}", "wb") as buffer:
        buffer.write(await file.read())

    file_path = f"app/uploads/extract/{file.filename}"

    #If category is type1
    logger.info('Parser')

    logger.info('Transformation Chain')
    load_image_chain = TransformChain(
        input_variables=["file_path"], output_variables=["image"], transform=load_image
    )

    vision_prompt = """Extrae los datos escritos a mano del informe. 
    El informe es semanal, cada grupo de columnas es un día de la semana, las filas son las tareas realizadas por cada día.
    Puede haber múltiples tareas por día, por lo que se debe extraer cada una de ellas.
    El campo código siempre empieza por I08 y después siguen más números.
    EL campo tiempo puede ser en minutos o en horas, por lo que se debe tener en cuenta.
    Pon el valor null en caso de que el campo esté vacío, no ivnventes los datos.
    """
    vision_chain = load_image_chain | report_model | report_parser
    #try:
    logger.info('Vision chain')
    return vision_chain.invoke(
        {"file_path": f"{file_path}", "prompt": vision_prompt}
    )
