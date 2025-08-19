import os
import glob
from dotenv import load_dotenv
import pypdf
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

print("Server is starting up and loading models...")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

genai.configure(api_key=api_key)


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection(name="local_rag_collection")

llm_model = genai.GenerativeModel("gemini-1.5-flash")

print("Models and Database loaded successfully. Server is ready to handle requests.")

app = FastAPI(
    title="Local RAG QnA System API",
    description="An API for queriying documents using a RAG pipeline with Google Gemini.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

def get_rag_answer(query: str) -> str:
    """
    Performs the RAG process: embedding, retrieval, and generation.
    """
    # 1. Create an embedding for the user's query
    query_embedding = embedding_model.encode(query).tolist()

    # 2. Query ChromaDB to find the most relevant document chunks
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    
    # 3. Combine the retrieved chunks into a single context string
    context = "\n---\n".join(results['documents'][0])
    
    # 4. Create the prompt for the LLM
    prompt_template = f"""
    CONTEXT:
    {context}
    
    QUESTION:
    {query}
    
    INSTRUCTIONS:
    Answer the user's QUESTION based ONLY on the provided CONTEXT.
    If the context does not contain the answer, state that you cannot answer based on the provided documents.
    Be concise and helpful.
    """
    
    # 5. Send the prompt to the Gemini API and get the response
    try:
        response = llm_model.generate_content(prompt_template)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the answer: {e}"
    
@app.post("/query", response_model=QueryResponse, tags=["RAG Query"])
async def query_endpoint(request: QueryRequest):
    """
    Receives a question, processes it through the RAG pipeline, and returns the answer.
    """
    answer = get_rag_answer(request.question)
    return {"answer": answer}

@app.get("/", tags=["Health Check"])
async def root():
    """A simple health check endpoint."""
    return {"message": "API is running. Send POST requests to /query"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

# # ---- DOCUMENT LOADING AND SPLITTING ----
# def load_documents(directory_path):
#     """Load text from all pdf files in a specified directory."""
#     documents = []
#     for item_path in glob.glob(os.path.join(directory_path, "*.pdf")):
#         print(f"Loading document: {os.path.basename(item_path)}")
#         try:
#             pdf_reader = pypdf.PdfReader(item_path)
#             text = ""
#             for page in pdf_reader.pages:
#                 page_text = page.extract_text()
#                 if page_text:
#                     text += page_text
#             documents.append({"content": text.strip(), "source": os.path.basename(item_path)})
#         except Exception as e:
#             print(f"Error Reading {item_path}: {e}")
#     print(f"Successfully loaded {len(documents)} documents.")
#     return documents

# def split_documents_into_chunks(documents, chunk_size=1000, chunk_overlap=200):
#     """Splits the text of loaded documents into smaller, overlapping chunks."""
#     all_chunks = []
#     for doc in documents:
#         text = doc["content"]
#         start_index = 0
#         while start_index < len(text):
#             end_index = start_index + chunk_size
#             chunk_content = text[start_index:end_index]
#             all_chunks.append({
#                 "content": chunk_content,
#                 "source": doc["source"]            
#             })
#             start_index += chunk_size - chunk_overlap
#     print(f"Split documents into {len(all_chunks)} chunks.")
#     return all_chunks

# def get_or_create_vector_db(chunks, embedding_model):
#     """Creates or loads a ChromaDB vector database form the document chunks."""
#     db_path = "chroma_db"
#     client = chromadb.PersistentClient(path=db_path)

#     collection_name = "local_rag_collection"
#     collection = client.get_or_create_collection(name=collection_name)

#     if collection.count() > 0:
#         print(f"Collection '{collection_name}' already exists and contains {collection.count()} documents. Skipping embedding.")
#         return collection
    
#     print(f"Creating embeddings for chunks. This may take a moment...")

#     ids = [str(i) for i in range(len(chunks))]
#     documents_to_embed = [chunk['content'] for chunk in chunks]

#     embeddings = embedding_model.encode(documents_to_embed, show_progress_bar=True)

#     print("Adding documents and embeddings to the collection...")

#     collection.add(
#         embeddings=embeddings.tolist(),
#         documents=documents_to_embed,
#         metadatas=[{'source': chunk['source']} for chunk in chunks],
#         ids=ids
#     )

#     print(f"Successfully added {len(ids)} documents to the '{collection_name}' collection.")
#     return collection

# def query_rag(query, collection, embedding_model, llm_model):
#     """Queries the RAG system."""
#     query_embedding = embedding_model.encode(query).tolist()

#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=3
#     )

#     context = "\n---\n".join(results['documents'][0])

#     prompt_template = f"""
#     CONTEXT:
#     {context}

#     QUESTION:
#     {query}

#     INSTRUCTIONS:
#     Answer the user's QUESTION based ONLY on the provided CONTEXT.
#     If the context does not contain the answer, state that you cannot answe.
#     Be conciss and helpful.
#     """

#     print("\n--- Sending Prompt to LLM ---")

#     try:
#         response = llm_model.generate_content(prompt_template)
#         return response.text
#     except Exception as e:
#         return f"An error occurred while querying the LLM: {e}"

# def main():
#     """The main function to run RAG pipeline"""
#     load_dotenv()
#     api_key = os.getenv("GOOGLE_API_KEY")
#     if not api_key:
#         print("Error: GOOGLE_API_KEY not found in .env file.")
#         return
#     # print("API Key loaded.")
#     genai.configure(api_key=api_key)

#     print("Loading sentence transformer model...")
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     print("Model loaded successfully.")


#     docs = load_documents("documents")
#     if not docs:
#         print("No documents found in the 'documents' folder. Please add a PDF.")
#         return
    
#     chunks = split_documents_into_chunks(docs)
#     # if chunks:
#     #     print("\n--- Verification ---")
#     #     print(f"Total Chunks Created: {len(chunks)}")
#     #     print(f"Sample Chunk 1:")
#     #     print(chunks[0]["content"])
#     #     print(f"(Source: { chunks[0]['source'] })")


#     collection = get_or_create_vector_db(chunks, embedding_model)

#     # print("\n--- Verification ---")
#     # print(f"ChromaDB collection '{collection.name}' is ready")
#     # print(f"Total documents in collection: {collection.count()}")

#     llm_model = genai.GenerativeModel('gemini-1.5-flash')

#     question = "Tell me about Abdul Khaleel"

#     print(f"\n--- Starting RAG Query ---")
#     print(f"Question: {question}")

#     answer = query_rag(question, collection, embedding_model, llm_model)

#     print("\n--- RAG System Answer ---")
#     print(answer)


# if __name__ == "__main__":
#     main()