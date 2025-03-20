import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import threading
import queue
import time

outputMaxLength = 200   # Setting the desired output length
MAX_BATCH_SIZE = 4      # Setting the desired maximum number of requests per batch
MAX_WAITING_TIME = 2    # Setting the desired maximum time (in seconds) to wait for a batch

documents = [           # Defining some documents in memory
    "Cats are small furry carnivores that are often kept as pets.",
    "Dogs are domesticated mammals, not natural wild animals.",
    "Hummingbirds can hover in mid-air by rapidly flapping their wings.",
    "Parrots are intelligent birds known for mimicking human speech.",
    "Rabbits are herbivorous mammals that often live in burrows.",
    "Goldfish are small freshwater fish commonly kept in bowls or tanks.",
    "Hamsters are small rodents frequently kept as household pets.",
    "Turtles are reptiles with hard shells, often kept as low-maintenance pets.",
    "Some birds, like pigeons and doves, are used in ceremonies and racing.",
    "Ferrets are playful, carnivorous mammals often adopted as exotic pets.",
    "Dogs are known for their loyalty and are trained for companionship, guarding, and assistance.",
    "Cats exhibit independent behavior and are often more solitary than dogs.",
    "Some animals, like hawks and falcons, are used in falconry.",
    "Koi fish are ornamental varieties of the common carp and symbolize good luck.",
    "Guinea pigs are sociable and vocal rodents originally from South America.",
]

app = FastAPI()

# 1. Load embedding model
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME)

# Basic Chat LLM
#chat_pipeline = pipeline("text-generation", model="facebook/opt-125m")
# Note: try this 1.5B model if you got enough GPU memory
chat_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")

# Initializing request queue
request_queue = queue.Queue()

# Initializing background thread
def process_batch():
    while True:
        batch = []
        futures = []
        while len(batch) < MAX_BATCH_SIZE and not request_queue.empty():
            request = request_queue.get()
            future = concurrent.futures.Future()  # Create a future object
            batch.append(request)
            futures.append(future)
            
        if batch:
            queries = [request["query"] for request in batch]
            results = [rag_pipeline(query) for query in queries]
            
            for result, future in zip(results, futures):
                future.set_result(result)  # Set result asynchronously
            
        time.sleep(MAX_WAITING_TIME)

# Start the background thread
thread = threading.Thread(target=process_batch, daemon=True)
thread.start()

def get_embedding(text: str) -> np.ndarray:
    """Compute a simple average-pool embedding."""
    inputs = embed_tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

# Precompute document embeddings
doc_embeddings = np.vstack([get_embedding(doc) for doc in documents])

### You may want to use your own top-k retrieval method (task 1)
def retrieve_top_k(query_emb: np.ndarray, k: int = 2) -> list:
    """Retrieve top-k docs via dot-product similarity."""
    sims = doc_embeddings @ query_emb.T
    top_k_indices = np.argsort(sims.ravel())[::-1][:k]
    return [documents[i] for i in top_k_indices]

def rag_pipeline(query: str, k: int = 2) -> str:
    # Step 1: Input embedding
    query_emb = get_embedding(query)
    
    # Step 2: Retrieval
    retrieved_docs = retrieve_top_k(query_emb, k)
    
    # Construct the prompt from query + retrieved docs
    context = "\n".join(retrieved_docs)
    prompt = f"Question: {query}\nContext:\n{context}\nAnswer:"
    
    # Step 3: LLM Output
    generated = chat_pipeline(prompt, max_length=outputMaxLength, do_sample=True)[0]["generated_text"]
    return generated

# Define request model
class QueryRequest(BaseModel):
    query: str
    k: int = 2

@app.post("/rag")
def predict(payload: QueryRequest):
    result = rag_pipeline(payload.query, payload.k)
    
    return {
        "query": payload.query,
        "result": result,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)