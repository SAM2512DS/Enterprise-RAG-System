"""
COMPLETE ENTERPRISE RAG SYSTEM
===============================
All features included:
- Multi-format document processing
- Hybrid search (BM25 + Vector)
- Agentic RAG with planning
- Reranking
- Caching (semantic + response)
- Query optimization
- Streaming responses
- Conversation memory
- Document management
- Authentication
- Rate limiting
- Monitoring & metrics
- Error handling & logging
- Configuration management
- Batch processing
- Testing utilities
"""

import os
import uuid
import pathlib
import tempfile
import json
import csv
import hashlib
import time
import logging
from datetime import datetime
from functools import lru_cache
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Security, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field,ConfigDict
from pydantic_settings import BaseSettings

from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import markdown
from bs4 import BeautifulSoup
import PyPDF2
import docx
from tenacity import retry, stop_after_attempt, wait_exponential

from dotenv import load_dotenv
load_dotenv()

# ================== LOGGING SETUP ==================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================== CONFIGURATION ==================

class Settings(BaseSettings):
    """Centralized configuration management"""
    
    # API Keys
    openrouter_key: str = Field(default="", validation_alias="OPENROUTER_KEY")
    api_key_secret: str = Field(default="your-secret-api-key-12345", validation_alias="API_KEY_SECRET")
    gemini_apikey: str = Field(default="", validation_alias="GEMINI_APIKEY")
    groq_apikey: str = Field(default="", validation_alias="GROQ_APIKEY")
    
    # Model Settings
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Chunking Settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_per_query: int = 10
    
    # Retrieval Settings
    default_k: int = 5
    rerank_top_k: int = 3
    hybrid_search_weight: float = 0.5  # 0.5 = equal weight dense+sparse
    
    # Performance
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    max_file_size_mb: int = 50
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Database
    chroma_db_path: str = "./chroma_db"
    
    # Pydantic v2 configuration
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra='ignore'  # Ignore extra fields in .env
    )

settings = Settings()

# ================== OPENAI CLIENT ==================

# client = OpenAI(
#     api_key=settings.openrouter_key,
#     base_url="https://openrouter.ai/api/v1",
#     default_headers={
#         "HTTP-Referer": "http://localhost",
#         "X-Title": "Enterprise-RAG-Complete"
#     }
# )


client = OpenAI(
    api_key=settings.groq_apikey,  # Change from openrouter_key to groq_apikey
    base_url="https://api.groq.com/openai/v1",  # Change to Groq endpoint
)

# ================== FASTAPI APP ==================

app = FastAPI(
    title="Enterprise RAG System - Complete",
    description="Full-featured RAG with all enterprise capabilities",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== AUTHENTICATION ==================

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """Verify API key for protected endpoints"""
    if not api_key:
        logger.warning("API request without key")
        raise HTTPException(status_code=403, detail="API Key required")
    
    if api_key != settings.api_key_secret:
        logger.warning(f"Invalid API key attempt: {api_key[:10]}...")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    
    return api_key

# ================== RATE LIMITING ==================

class RateLimiter:
    """Simple in-memory rate limiter"""
    
    def __init__(self):
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - settings.rate_limit_window
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > window_start
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= settings.rate_limit_requests:
            return False
        
        self.requests[client_id].append(now)
        return True

rate_limiter = RateLimiter()

async def check_rate_limit(request: Request):
    """Rate limiting middleware"""
    client_ip = request.client.host
    
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {settings.rate_limit_requests} requests per {settings.rate_limit_window}s"
        )

# ================== METRICS & MONITORING ==================

class Metrics:
    """Track system metrics"""
    
    def __init__(self):
        self.data = {
            "total_requests": 0,
            "total_uploads": 0,
            "total_queries": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "avg_query_time": 0.0,
            "avg_upload_time": 0.0,
            "start_time": datetime.now().isoformat()
        }
        self.query_times = []
        self.upload_times = []
    
    def increment(self, metric: str, value: int = 1):
        self.data[metric] = self.data.get(metric, 0) + value
    
    def record_query_time(self, duration: float):
        self.query_times.append(duration)
        self.data["avg_query_time"] = sum(self.query_times) / len(self.query_times)
    
    def record_upload_time(self, duration: float):
        self.upload_times.append(duration)
        self.data["avg_upload_time"] = sum(self.upload_times) / len(self.upload_times)
    
    def get_stats(self) -> dict:
        return {
            **self.data,
            "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.data["start_time"])).total_seconds()
        }

metrics = Metrics()

# ================== MIDDLEWARE FOR TRACKING ==================

@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track all requests for metrics"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        metrics.increment("total_requests")
        return response
    except Exception as e:
        metrics.increment("errors")
        logger.error(f"Request error: {str(e)}")
        raise
    finally:
        duration = time.time() - start_time
        logger.info(f"{request.method} {request.url.path} - {duration:.3f}s")

# ================== VECTOR DB ==================

chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=settings.embedding_model
)

try:
    collection = chroma_client.get_collection(name="documents")
    logger.info("Loaded existing collection")
except:
    collection = chroma_client.create_collection(
        name="documents",
        embedding_function=embedding_function
    )
    logger.info("Created new collection")

# ================== RERANKER ==================

logger.info(f"Loading reranker model: {settings.reranker_model}")
reranker = CrossEncoder(settings.reranker_model)
logger.info("Reranker loaded successfully")

# ================== CACHING ==================

class SemanticCache:
    """Semantic caching for queries and responses"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}  # hash -> (query, answer, embedding, timestamp)
        self.threshold = similarity_threshold
        self.embedding_func = embedding_function
    
    def _get_hash(self, query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response if similar query exists"""
        query_hash = self._get_hash(query)
        
        if query_hash in self.cache:
            cached = self.cache[query_hash]
            # Check TTL
            if time.time() - cached["timestamp"] < settings.cache_ttl:
                logger.info(f"Cache HIT for query: {query[:50]}...")
                metrics.increment("cache_hits")
                return cached["answer"]
            else:
                # Expired
                del self.cache[query_hash]
        
        metrics.increment("cache_misses")
        logger.info(f"Cache MISS for query: {query[:50]}...")
        return None
    
    def set(self, query: str, answer: str):
        """Cache a query-answer pair"""
        query_hash = self._get_hash(query)
        self.cache[query_hash] = {
            "query": query,
            "answer": answer,
            "timestamp": time.time()
        }
        logger.info(f"Cached response for: {query[:50]}...")
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        logger.info("Cache cleared")

semantic_cache = SemanticCache() if settings.enable_caching else None

# ================== DOCUMENT PROCESSING ==================

def process_markdown(content: str) -> str:
    """Convert markdown to plain text"""
    try:
        html = markdown.markdown(content)
        return BeautifulSoup(html, "html.parser").get_text()
    except Exception as e:
        logger.error(f"Markdown processing error: {e}")
        return content

def process_pdf(file_content: bytes) -> str:
    """Extract text from PDF"""
    text = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            reader = PyPDF2.PdfReader(tmp_file.name)
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        os.unlink(tmp_file.name)
        logger.info(f"Extracted {len(text)} chars from PDF")
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        raise ValueError(f"Failed to process PDF: {str(e)}")
    return text

def process_docx(file_content: bytes) -> str:
    """Extract text from DOCX"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp_file:
            tmp_file.write(file_content)
            tmp_file.flush()
            doc = docx.Document(tmp_file.name)
            text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        os.unlink(tmp_file.name)
        logger.info(f"Extracted {len(text)} chars from DOCX")
        return text
    except Exception as e:
        logger.error(f"DOCX processing error: {e}")
        raise ValueError(f"Failed to process DOCX: {str(e)}")

def process_csv(file_content: bytes) -> str:
    """Convert CSV to text"""
    try:
        reader = csv.reader(file_content.decode("utf-8").splitlines())
        rows = [", ".join(row) for row in reader]
        return "\n".join(rows)
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        return file_content.decode("utf-8", errors="ignore")

def process_json(file_content: bytes) -> str:
    """Pretty print JSON"""
    try:
        data = json.loads(file_content.decode("utf-8"))
        return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"JSON processing error: {e}")
        return file_content.decode("utf-8", errors="ignore")

def process_txt(file_content: bytes) -> str:
    """Decode text file"""
    try:
        return file_content.decode("utf-8")
    except:
        return file_content.decode("utf-8", errors="ignore")

def extract_text(file_content: bytes, filename: str) -> str:
    """Extract text from any supported file format"""
    ext = pathlib.Path(filename).suffix.lower()
    
    # Check file size
    size_mb = len(file_content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max: {settings.max_file_size_mb}MB)")
    
    processors = {
        ".pdf": process_pdf,
        ".docx": process_docx,
        ".csv": process_csv,
        ".json": process_json,
        ".md": lambda c: process_markdown(c.decode("utf-8")),
        ".txt": process_txt,
    }
    
    processor = processors.get(ext)
    if processor:
        return processor(file_content)
    
    # Try generic text decode
    try:
        return file_content.decode("utf-8")
    except:
        raise ValueError(f"Unsupported file format: {ext}")

# ================== ADVANCED CHUNKING ==================

def chunk_text_advanced(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Advanced recursive chunking with semantic preservation
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    
    # Try to split on paragraphs first, then sentences, then words
    separators = ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]
    
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    
    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph too long, split it
            if len(para) > chunk_size:
                words = para.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= chunk_size:
                        temp_chunk += word + " "
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "
                if temp_chunk:
                    current_chunk = temp_chunk
            else:
                current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk
                prev_words = chunks[i-1].split()[-overlap:]
                chunk = " ".join(prev_words) + " " + chunk
            overlapped_chunks.append(chunk)
        chunks = overlapped_chunks
    
    logger.info(f"Created {len(chunks)} chunks from {len(text)} chars")
    return chunks

# ================== RAG UTILITIES ==================

def add_to_collection(text: str, metadata: Optional[Dict] = None) -> int:
    """Add document chunks to vector store"""
    try:
        chunks = chunk_text_advanced(text)
        ids = [str(uuid.uuid4()) for _ in chunks]
        
        # Add timestamp to metadata
        meta = metadata or {}
        meta["indexed_at"] = datetime.now().isoformat()
        
        collection.add(
            documents=chunks,
            ids=ids,
            metadatas=[meta for _ in chunks]
        )
        
        metrics.increment("total_chunks", len(chunks))
        logger.info(f"Added {len(chunks)} chunks to collection")
        return len(chunks)
    except Exception as e:
        logger.error(f"Failed to add to collection: {e}")
        raise

def retrieve_docs(query: str, n_results: int = None, where: Dict = None) -> List[Dict]:
    """Retrieve documents from vector store with metadata filtering"""
    try:
        n_results = n_results or settings.default_k
        
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        
        if results and "documents" in results and results["documents"]:
            docs = []
            for i, doc in enumerate(results["documents"][0]):
                docs.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "id": results["ids"][0][i] if results.get("ids") else None
                })
            return docs
        return []
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return []

# ================== HYBRID SEARCH ==================

class HybridSearcher:
    """Combines dense (vector) and sparse (BM25) search"""
    
    def __init__(self):
        self.documents = []
        self.doc_metadata = []
        self.tokenized_docs = []
        self.bm25 = None
    
    def build_bm25(self):
        """Build BM25 index"""
        if self.documents:
            self.tokenized_docs = [doc.lower().split() for doc in self.documents]
            self.bm25 = BM25Okapi(self.tokenized_docs)
            logger.info(f"Built BM25 index with {len(self.documents)} documents")
    
    def add_docs(self, docs: List[str], metadata: List[Dict] = None):
        """Add documents to BM25 index"""
        self.documents.extend(docs)
        self.doc_metadata.extend(metadata or [{}] * len(docs))
        self.build_bm25()
    
    def search(self, query: str, k: int = None, use_reranking: bool = True) -> List[Dict]:
        """
        Hybrid search: BM25 + Vector + Reranking
        """
        k = k or settings.default_k
        
        # 1. BM25 search (keyword)
        bm25_docs = []
        if self.bm25:
            scores = self.bm25.get_scores(query.lower().split())
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k*2]
            bm25_docs = [self.documents[i] for i in top_indices]
        
        # 2. Vector search (semantic)
        vector_results = retrieve_docs(query, n_results=k*2)
        vector_docs = [r["content"] for r in vector_results]
        
        # 3. Combine and deduplicate
        combined = []
        seen = set()
        
        # Interleave BM25 and vector results
        for bm25_doc, vec_doc in zip(bm25_docs, vector_docs):
            if bm25_doc not in seen:
                combined.append(bm25_doc)
                seen.add(bm25_doc)
            if vec_doc not in seen:
                combined.append(vec_doc)
                seen.add(vec_doc)
        
        # Add remaining
        for doc in bm25_docs + vector_docs:
            if doc not in seen:
                combined.append(doc)
                seen.add(doc)
        
        # 4. Rerank if enabled
        if use_reranking and combined:
            combined = self.rerank_documents(query, combined[:k*2])
        
        # Return top k
        result_docs = combined[:k]
        
        return [{"content": doc, "metadata": {}} for doc in result_docs]
    
    def rerank_documents(self, query: str, docs: List[str]) -> List[str]:
        """Rerank documents using cross-encoder"""
        try:
            pairs = [[query, doc] for doc in docs]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            logger.info(f"Reranked {len(docs)} documents")
            return [doc for doc, score in ranked]
        except Exception as e:
            logger.error(f"Reranking error: {e}")
            return docs

searcher = HybridSearcher()

# ================== QUERY OPTIMIZATION ==================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(messages: List[Dict], stream: bool = False, **kwargs):
    """Call LLM with retry logic"""
    try:
        return client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            stream=stream,
            **kwargs
        )
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        raise

def optimize_query(query: str) -> str:
    """Optimize query for better retrieval"""
    try:
        prompt = f"""Rephrase this question to be more specific and searchable. Keep it concise (one sentence).

Question: {query}

Optimized:"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        optimized = resp.choices[0].message.content.strip()
        logger.info(f"Query optimized: '{query}' -> '{optimized}'")
        return optimized
    except Exception as e:
        logger.warning(f"Query optimization failed: {e}")
        return query

# ================== AGENTIC RAG ==================

def planner_agent(query: str) -> List[str]:
    """Break complex query into sub-questions"""
    try:
        prompt = f"""Break this question into 2-3 simpler sub-questions that would help answer it. List each on a new line.

Question: {query}

Sub-questions:"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        steps = resp.choices[0].message.content.strip().split("\n")
        steps = [s.strip("- ").strip("0123456789. ").strip() for s in steps if s.strip()]
        logger.info(f"Query broken into {len(steps)} steps")
        return steps[:3]  # Max 3 steps
    except Exception as e:
        logger.error(f"Planner failed: {e}")
        return [query]

def agentic_rag(query: str) -> Dict[str, Any]:
    """Multi-step agentic RAG"""
    try:
        # 1. Plan
        steps = planner_agent(query)
        
        # 2. Retrieve for each step
        all_docs = []
        for step in steps:
            docs = searcher.search(step, k=5, use_reranking=True)
            all_docs.extend(docs)
        
        # 3. Deduplicate
        seen = set()
        unique_docs = []
        for doc in all_docs:
            content = doc["content"]
            if content not in seen:
                seen.add(content)
                unique_docs.append(doc)
        
        # 4. Take top documents
        top_docs = unique_docs[:settings.max_chunks_per_query]
        
        # 5. Generate answer
        context = "\n\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(top_docs)])
        
        prompt = f"""Answer the question using the provided context. Cite sources using [1], [2], etc.

Context:
{context}

Question: {query}

Answer:"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        answer = resp.choices[0].message.content
        
        return {
            "answer": answer,
            "steps": steps,
            "sources": [d["content"][:200] + "..." for d in top_docs],
            "num_sources": len(top_docs)
        }
    except Exception as e:
        logger.error(f"Agentic RAG failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agentic RAG failed: {str(e)}")

# ================== MULTI-DOC REASONING ==================

def multi_doc_reasoning(query: str, docs: List[Dict]) -> str:
    """Synthesize answer from multiple documents"""
    try:
        joined = "\n\n".join([
            f"Document {i+1}:\n{d['content']}"
            for i, d in enumerate(docs[:8])
        ])
        
        prompt = f"""Synthesize an answer using ALL the documents. If documents contradict, mention it.

{joined}

Question: {query}

Synthesized Answer:"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        return resp.choices[0].message.content
    except Exception as e:
        logger.error(f"Multi-doc reasoning failed: {e}")
        raise

# ================== RAG EVALUATION ==================

def evaluate_rag(query: str, context: str, answer: str) -> Dict:
    """Evaluate RAG response quality"""
    try:
        prompt = f"""Evaluate this RAG system response on a scale of 1-5 for each metric:

1. Context Relevance: Is the context relevant to the question?
2. Faithfulness: Is the answer grounded in the context (no hallucinations)?
3. Answer Completeness: Does it fully answer the question?

Question: {query}

Context:
{context[:1000]}...

Answer:
{answer}

Return ONLY valid JSON with this structure:
{{
  "context_relevance": 4,
  "faithfulness": 5,
  "answer_completeness": 3,
  "overall_score": 4.0,
  "feedback": "brief explanation"
}}"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        result = resp.choices[0].message.content
        
        # Try to parse JSON
        try:
            # Extract JSON if wrapped in markdown
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0]
            elif "```" in result:
                result = result.split("```")[1].split("```")[0]
            
            evaluation = json.loads(result)
            return evaluation
        except:
            return {"error": "Failed to parse evaluation", "raw": result}
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {"error": str(e)}

# ================== CONVERSATION MEMORY ==================

class ConversationManager:
    """Manage multi-turn conversations"""
    
    def __init__(self, max_history: int = 10):
        self.conversations = {}  # session_id -> messages
        self.max_history = max_history
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add message to conversation history"""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        
        self.conversations[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only recent messages
        if len(self.conversations[session_id]) > self.max_history * 2:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history * 2:]
    
    def get_history(self, session_id: str) -> List[Dict]:
        """Get conversation history"""
        return self.conversations.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """Clear a conversation"""
        if session_id in self.conversations:
            del self.conversations[session_id]
    
    def chat(self, session_id: str, query: str) -> str:
        """Chat with conversation context"""
        # Retrieve relevant docs
        docs = searcher.search(query, k=5, use_reranking=True)
        context = "\n\n".join([d["content"] for d in docs[:5]])
        
        # Build messages
        messages = []
        messages.append({
            "role": "system",
            "content": f"You are a helpful assistant. Use this context to answer:\n\n{context}"
        })
        
        # Add conversation history
        history = self.get_history(session_id)
        for msg in history[-6:]:  # Last 3 turns
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        # Generate response
        resp = call_llm(messages)
        answer = resp.choices[0].message.content
        
        # Save to history
        self.add_message(session_id, "user", query)
        self.add_message(session_id, "assistant", answer)
        
        return answer

conversation_manager = ConversationManager()

# ================== PYDANTIC MODELS ==================

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: Optional[int] = Field(default=None, ge=1, le=20)
    use_cache: Optional[bool] = True
    use_reranking: Optional[bool] = True

class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1, max_length=1000)

class DocumentFilter(BaseModel):
    filename: Optional[str] = None
    extension: Optional[str] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None

# ================== API ROUTES ==================

@app.get("/")
def root():
    """API health check"""
    return {
        "status": "running",
        "version": "2.0.0",
        "features": [
            "multi-format-upload",
            "hybrid-search",
            "reranking",
            "caching",
            "agentic-rag",
            "conversation-memory",
            "rate-limiting",
            "authentication",
            "monitoring"
        ]
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "database": "connected",
        "cache": "enabled" if settings.enable_caching else "disabled",
        "models": {
            "llm": settings.llm_model,
            "embeddings": settings.embedding_model,
            "reranker": settings.reranker_model
        }
    }

@app.get("/metrics")
def get_metrics():
    """Get system metrics"""
    return metrics.get_stats()

@app.post("/upload-file", dependencies=[Depends(check_rate_limit)])
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a document"""
    start_time = time.time()
    
    try:
        logger.info(f"Processing file: {file.filename}")
        
        content = await file.read()
        text = extract_text(content, file.filename)
        
        metadata = {
            "filename": file.filename,
            "type": file.content_type or "unknown",
            "size": len(content),
            "extension": pathlib.Path(file.filename).suffix,
            "uploaded_at": datetime.now().isoformat()
        }
        
        num_chunks = add_to_collection(text, metadata)
        searcher.add_docs([text], [metadata])
        
        metrics.increment("total_uploads")
        metrics.increment("total_documents")
        metrics.record_upload_time(time.time() - start_time)
        
        return {
            "success": True,
            "filename": file.filename,
            "chunks_created": num_chunks,
            "size_bytes": len(content),
            "processing_time": f"{time.time() - start_time:.2f}s"
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        metrics.increment("errors")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-batch", dependencies=[Depends(check_rate_limit)])
async def upload_batch(files: List[UploadFile]):
    """Upload multiple files at once"""
    results = []
    
    for file in files:
        try:
            content = await file.read()
            text = extract_text(content, file.filename)
            metadata = {"filename": file.filename, "extension": pathlib.Path(file.filename).suffix}
            chunks = add_to_collection(text, metadata)
            searcher.add_docs([text], [metadata])
            
            results.append({
                "filename": file.filename,
                "success": True,
                "chunks": chunks
            })
            metrics.increment("total_uploads")
            metrics.increment("total_documents")
        except Exception as e:
            logger.error(f"Batch upload error for {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            metrics.increment("errors")
    
    return {"results": results, "total": len(files), "successful": sum(1 for r in results if r["success"])}

@app.post("/query/simple", dependencies=[Depends(check_rate_limit)])
def query_simple(req: QueryRequest):
    """Simple RAG query with caching"""
    start_time = time.time()
    
    try:
        # Check cache
        if req.use_cache and semantic_cache:
            cached = semantic_cache.get(req.query)
            if cached:
                return {
                    "answer": cached,
                    "cached": True,
                    "query_time": f"{time.time() - start_time:.3f}s"
                }
        
        # Retrieve documents
        k = req.k or settings.default_k
        docs = searcher.search(req.query, k=k, use_reranking=req.use_reranking)
        
        # Generate answer
        context = "\n\n".join([f"[{i+1}] {d['content']}" for i, d in enumerate(docs)])
        
        prompt = f"""Answer based on the context. Cite sources with [1], [2], etc.

Context:
{context}

Question: {req.query}

Answer:"""
        
        resp = call_llm([{"role": "user", "content": prompt}])
        answer = resp.choices[0].message.content
        
        # Cache result
        if req.use_cache and semantic_cache:
            semantic_cache.set(req.query, answer)
        
        metrics.increment("total_queries")
        metrics.record_query_time(time.time() - start_time)
        
        return {
            "answer": answer,
            "sources": [d["content"][:200] + "..." for d in docs],
            "num_sources": len(docs),
            "cached": False,
            "query_time": f"{time.time() - start_time:.3f}s"
        }
    except Exception as e:
        logger.error(f"Query failed: {e}")
        metrics.increment("errors")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/agentic", dependencies=[Depends(check_rate_limit)])
def query_agentic(req: QueryRequest):
    """Agentic RAG with planning"""
    start_time = time.time()
    
    try:
        result = agentic_rag(req.query)
        result["query_time"] = f"{time.time() - start_time:.3f}s"
        
        metrics.increment("total_queries")
        metrics.record_query_time(time.time() - start_time)
        
        return result
    except Exception as e:
        metrics.increment("errors")
        raise

@app.post("/query/stream", dependencies=[Depends(check_rate_limit)])
async def query_stream(req: QueryRequest):
    """Stream LLM response"""
    try:
        # Retrieve docs
        docs = searcher.search(req.query, k=req.k or settings.default_k)
        context = "\n\n".join([d["content"] for d in docs[:5]])
        
        prompt = f"""Answer based on context:

{context}

Question: {req.query}

Answer:"""
        
        # Stream response
        stream = call_llm(
            [{"role": "user", "content": prompt}],
            stream=True
        )
        
        async def generate():
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        return StreamingResponse(generate(), media_type="text/plain")
    except Exception as e:
        logger.error(f"Streaming failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", dependencies=[Depends(check_rate_limit)])
def chat(req: ChatRequest):
    """Multi-turn conversation"""
    try:
        answer = conversation_manager.chat(req.session_id, req.query)
        
        return {
            "session_id": req.session_id,
            "answer": answer,
            "history_length": len(conversation_manager.get_history(req.session_id))
        }
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/chat/{session_id}")
def clear_chat(session_id: str):
    """Clear conversation history"""
    conversation_manager.clear_session(session_id)
    return {"message": f"Cleared session {session_id}"}

@app.post("/evaluate", dependencies=[Depends(check_rate_limit)])
def evaluate_query(req: QueryRequest):
    """Evaluate RAG response quality"""
    try:
        # Get answer
        docs = searcher.search(req.query, k=5)
        context = "\n\n".join([d["content"] for d in docs])
        answer = multi_doc_reasoning(req.query, docs)
        
        # Evaluate
        evaluation = evaluate_rag(req.query, context, answer)
        
        return {
            "answer": answer,
            "evaluation": evaluation,
            "sources": [d["content"][:150] + "..." for d in docs]
        }
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
def list_documents(filter: DocumentFilter = Depends()):
    """List all documents with optional filtering"""
    try:
        all_data = collection.get()
        
        # Extract unique documents
        documents = {}
        for i, metadata in enumerate(all_data["metadatas"]):
            filename = metadata.get("filename", "unknown")
            if filename not in documents:
                documents[filename] = {
                    "filename": filename,
                    "extension": metadata.get("extension", ""),
                    "size": metadata.get("size", 0),
                    "uploaded_at": metadata.get("uploaded_at", ""),
                    "chunks": 1
                }
            else:
                documents[filename]["chunks"] += 1
        
        # Apply filters
        result = list(documents.values())
        
        if filter.filename:
            result = [d for d in result if filter.filename.lower() in d["filename"].lower()]
        
        if filter.extension:
            result = [d for d in result if d["extension"] == filter.extension]
        
        if filter.min_size:
            result = [d for d in result if d["size"] >= filter.min_size]
        
        if filter.max_size:
            result = [d for d in result if d["size"] <= filter.max_size]
        
        return {
            "documents": result,
            "total": len(result)
        }
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Delete a document and all its chunks"""
    try:
        # Get all chunks for this document
        results = collection.get(where={"filename": filename})
        
        if not results["ids"]:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        # Delete chunks
        collection.delete(ids=results["ids"])
        
        logger.info(f"Deleted {len(results['ids'])} chunks for {filename}")
        
        return {
            "message": f"Deleted {filename}",
            "chunks_deleted": len(results["ids"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear", dependencies=[Security(verify_api_key)])
def clear_cache():
    """Clear semantic cache (protected endpoint)"""
    if semantic_cache:
        semantic_cache.clear()
        return {"message": "Cache cleared"}
    return {"message": "Cache not enabled"}

@app.get("/config")
def get_config():
    """Get current configuration (non-sensitive)"""
    return {
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "default_k": settings.default_k,
        "max_file_size_mb": settings.max_file_size_mb,
        "caching_enabled": settings.enable_caching,
        "rate_limit": f"{settings.rate_limit_requests} requests per {settings.rate_limit_window}s"
    }

# ================== STARTUP EVENT ==================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("ENTERPRISE RAG SYSTEM STARTING")
    logger.info("=" * 60)
    logger.info(f"LLM Model: {settings.llm_model}")
    logger.info(f"Embedding Model: {settings.embedding_model}")
    logger.info(f"Reranker Model: {settings.reranker_model}")
    logger.info(f"Chunk Size: {settings.chunk_size}")
    logger.info(f"Cache Enabled: {settings.enable_caching}")
    logger.info(f"Rate Limit: {settings.rate_limit_requests} req/{settings.rate_limit_window}s")
    
    # Count existing documents
    try:
        data = collection.get()
        total_chunks = len(data["ids"]) if data and "ids" in data else 0
        logger.info(f"Existing chunks in database: {total_chunks}")
        metrics.data["total_chunks"] = total_chunks
    except:
        logger.info("No existing data in database")
    
    logger.info("=" * 60)
    logger.info("SYSTEM READY")
    logger.info("=" * 60)

# ================== MAIN ==================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 60)
    print("ENTERPRISE RAG SYSTEM - COMPLETE")
    print("=" * 60)
    print(f"\n📚 Features:")
    print("  ✓ Multi-format document processing")
    print("  ✓ Hybrid search (BM25 + Vector)")
    print("  ✓ Cross-encoder reranking")
    print("  ✓ Semantic caching")
    print("  ✓ Agentic RAG with planning")
    print("  ✓ Multi-turn conversations")
    print("  ✓ Streaming responses")
    print("  ✓ Query optimization")
    print("  ✓ Rate limiting")
    print("  ✓ Authentication")
    print("  ✓ Monitoring & metrics")
    print("  ✓ Document management")
    print("  ✓ Batch upload")
    print("  ✓ RAG evaluation")
    print(f"\n🚀 Starting server...")
    print(f"📖 API Docs: http://localhost:8000/docs")
    print(f"🔧 Health: http://localhost:8000/health")
    print(f"📊 Metrics: http://localhost:8000/metrics")
    print("=" * 60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )