from fastapi import FastAPI
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import os

app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

embeddings = OllamaEmbeddings(model="llama3")
vector_db = Chroma(persist_directory="./story_db", embedding_function=embeddings)
llm = OllamaLLM(model="llama3")

@app.get("/")
def home():
    return {"message": "ScriptFlow-AI Server is Online"}

@app.post("/add_lore")
async def add_lore(fact: str):   
    """Save a character trait or plot point to the Story Bible."""
    vector_db.add_texts([fact])  
    return {"message": "Lore saved to Story Bible"}

@app.post("/generate_with_context")
async def generate_with_context(prompt: str):
    """Retrieve relevant lore before generating the script."""
    # Search DB for the 2 most relevant past facts
    docs= vector_db.similarity_search(prompt, k=2)
    context = "\n".join([d.page_content for d in docs])
    
    # Create a combined prompt
    enriched_prompt = f"Context from Story Bible:\n{context}\n\nTask:{prompt}"
    
    response= llm.invoke(enriched_prompt)
    return {"script": response}