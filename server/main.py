from fastapi import FastAPI
from dotenv import load_dotenv
# from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
import os

load_dotenv()
app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# GEMINI SETUP 
gemini_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=gemini_key)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_key)

# OLLAMA SETUP
# ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# embeddings = OllamaEmbeddings(model="llama3")
# vector_db = Chroma(persist_directory="./story_db", embedding_function=embeddings)
# llm = OllamaLLM(model="llama3")

# Initialize Vector DB 
vector_db = Chroma(persist_directory="./story_db", embedding_function=embeddings)

@app.get("/")
def home():
    return {"message": "ScriptFlow-AI Server is Online"}

@app.post("/add_lore")
async def add_lore(fact: str):   
    # """Save a character trait or plot point to the Story Bible."""
    vector_db.add_texts([fact])  
    return {"message": "Lore saved to Story Bible"}

@app.post("/generate_with_context")
async def generate_with_context(prompt: str):
    # """Retrieve relevant lore before generating the script."""
    # Search DB for the 2 most relevant past facts
    docs= vector_db.similarity_search(prompt, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # Create a combined prompt
    enriched_prompt = f"Context from Story Bible:\n{context}\n\nTask:{prompt}"
    
    response= llm.invoke(enriched_prompt)
    return {"script": response}