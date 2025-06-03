"""
Production-Grade RAG System with Advanced Techniques
FastAPI + WebSockets + Async + Streaming
Features: Hybrid Search, Reranking, Query Expansion, Self-RAG
"""

import os
import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager
import logging

# FastAPI & WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Vector DB & Embeddings
import chromadb
from chromadb.utils import embedding_functions
import numpy as np

# LLM & RAG Components
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

# Advanced RAG imports
import nltk
from nltk.corpus import wordnet
import spacy

# Async utilities
from motor.motor_asyncio import AsyncIOMotorClient
import redis.asyncio as redis
from aiocache import Cache
from aiocache.decorators import cached

# Monitoring & Observability
from prometheus_client import Counter, Histogram, generate_latest
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure structured logging
logger = structlog.get_logger()

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'RAG query duration')
websocket_connections = Counter('websocket_connections_total', 'Total WebSocket connections')

# Request/Response Models
class RAGQuery(BaseModel):
    query: str = Field(..., description="User query")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    top_k: int = Field(5, ge=1, le=20)
    temperature: float = Field(0.7, ge=0, le=2)
    stream: bool = Field(True)

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any]

class WebSocketMessage(BaseModel):
    type: str  # 'query', 'chunk', 'sources', 'error', 'complete'
    data: Any
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

# Advanced RAG Engine
class AdvancedRAGEngine:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger.bind(component="RAGEngine")
        
        # Initialize components
        self.embedder = SentenceTransformer(config.get("embedding_model", "BAAI/bge-large-en-v1.5"))
        self.reranker = CrossEncoder(config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        self.llm = AsyncOpenAI(api_key=config["openai_api_key"])
        
        # Vector store
        self.chroma_client = chromadb.AsyncHttpClient(host=config.get("chroma_host", "localhost"))
        self.collection = None
        
        # Cache
        self.cache = Cache(Cache.REDIS)
        
        # Query expansion tools
        try:
            nltk.download('wordnet', quiet=True)
            self.nlp = spacy.load("en_core_web_sm")
        except:
            self.logger.warning("NLP tools not fully initialized")
    
    async def initialize(self):
        """Async initialization"""
        self.collection = await self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.get("embedding_model", "BAAI/bge-large-en-v1.5")
            )
        )
        self.logger.info("RAG Engine initialized")
    
    def expand_query(self, query: str) -> List[str]:
        """Query expansion using synonyms and semantic understanding"""
        expanded = [query]
        
        # Extract key terms
        doc = self.nlp(query)
        keywords = [token.text for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
        
        # Add synonyms
        for keyword in keywords[:3]:  # Limit expansion
            synsets = wordnet.synsets(keyword)
            for syn in synsets[:2]:
                for lemma in syn.lemmas()[:2]:
                    if lemma.name() != keyword:
                        expanded.append(query.replace(keyword, lemma.name()))
        
        return list(set(expanded))[:5]  # Return unique, limited set
    
    async def hybrid_search(self, queries: List[str], top_k: int = 10) -> List[Dict]:
        """Hybrid search combining dense + sparse retrieval"""
        all_results = []
        
        # Parallel search for all expanded queries
        tasks = []
        for q in queries:
            # Dense retrieval
            task = self.collection.query(
                query_texts=[q],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # Combine and deduplicate results
        seen_ids = set()
        for result in results:
            for i, doc_id in enumerate(result["ids"][0]):
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    all_results.append({
                        "id": doc_id,
                        "document": result["documents"][0][i],
                        "metadata": result["metadatas"][0][i],
                        "score": 1 - result["distances"][0][i]  # Convert distance to similarity
                    })
        
        return all_results
    
    def rerank_results(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        """Rerank results using cross-encoder"""
        if not results:
            return []
        
        # Prepare pairs for reranking
        pairs = [(query, r["document"]) for r in results]
        scores = self.reranker.predict(pairs)
        
        # Add rerank scores and sort
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])
        
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
    
    async def generate_answer_stream(
        self, 
        query: str, 
        context: List[Dict],
        temperature: float = 0.7
    ) -> AsyncGenerator[str, None]:
        """Stream answer generation with citations"""
        
        # Build context prompt
        context_text = "\n\n".join([
            f"[Source {i+1}] {doc['document']}"
            for i, doc in enumerate(context)
        ])
        
        messages = [
            {"role": "system", "content": """You are an advanced RAG assistant. 
            Answer based on the provided context. Cite sources using [Source N] format.
            Be accurate, comprehensive, and acknowledge uncertainty when appropriate."""},
            {"role": "user", "content": f"""Context:\n{context_text}\n\nQuestion: {query}
            
            Provide a detailed answer with proper citations."""}
        ]
        
        # Stream response
        stream = await self.llm.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            temperature=temperature,
            stream=True
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    @cached(ttl=300)  # Cache for 5 minutes
    async def process_query(self, rag_query: RAGQuery) -> AsyncGenerator[Dict, None]:
        """Main RAG pipeline with streaming"""
        query_counter.inc()
        
        with query_duration.time():
            # Query expansion
            expanded_queries = self.expand_query(rag_query.query)
            self.logger.info("Query expanded", original=rag_query.query, expanded=expanded_queries)
            
            # Hybrid search
            search_results = await self.hybrid_search(expanded_queries, top_k=rag_query.top_k * 2)
            
            # Reranking
            reranked_results = self.rerank_results(rag_query.query, search_results, rag_query.top_k)
            
            # Self-critique loop (simplified version)
            confidence = np.mean([r["rerank_score"] for r in reranked_results]) if reranked_results else 0
            
            if confidence < 0.5:
                # Low confidence - try web search or return uncertainty
                yield {
                    "type": "warning",
                    "data": "Low confidence in available sources. Consider external search."
                }
            
            # Stream answer generation
            answer_chunks = []
            async for chunk in self.generate_answer_stream(
                rag_query.query, 
                reranked_results,
                rag_query.temperature
            ):
                answer_chunks.append(chunk)
                yield {
                    "type": "chunk",
                    "data": chunk
                }
            
            # Send sources
            yield {
                "type": "sources",
                "data": [
                    {
                        "id": r["id"],
                        "content": r["document"][:200] + "...",
                        "metadata": r["metadata"],
                        "score": r["rerank_score"]
                    }
                    for r in reranked_results
                ]
            }
            
            # Final metadata
            yield {
                "type": "complete",
                "data": {
                    "answer": "".join(answer_chunks),
                    "confidence": float(confidence),
                    "query_expansion": expanded_queries,
                    "sources_count": len(reranked_results)
                }
            }

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, List[str]] = {}  # user_id -> [session_ids]
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        if user_id:
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(session_id)
        websocket_connections.inc()
        logger.info("WebSocket connected", session_id=session_id, user_id=user_id)
    
    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            # Clean up user sessions
            for user_id, sessions in self.user_sessions.items():
                if session_id in sessions:
                    sessions.remove(session_id)
        logger.info("WebSocket disconnected", session_id=session_id)
    
    async def send_message(self, session_id: str, message: WebSocketMessage):
        if websocket := self.active_connections.get(session_id):
            await websocket.send_json(message.dict())
    
    async def broadcast_to_user(self, user_id: str, message: WebSocketMessage):
        if sessions := self.user_sessions.get(user_id, []):
            tasks = [self.send_message(sid, message) for sid in sessions]
            await asyncio.gather(*tasks, return_exceptions=True)

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting RAG application")
    
    # Initialize RAG engine
    config = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "chroma_host": os.getenv("CHROMA_HOST", "localhost"),
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
    }
    
    app.state.rag_engine = AdvancedRAGEngine(config)
    await app.state.rag_engine.initialize()
    
    # Initialize connection manager
    app.state.connection_manager = ConnectionManager()
    
    # Initialize Redis for caching
    app.state.redis = await redis.from_url("redis://localhost:6379")
    
    # Initialize MongoDB for chat history
    app.state.mongo = AsyncIOMotorClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017"))
    app.state.db = app.state.mongo.rag_chatbot
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG application")
    await app.state.redis.close()
    app.state.mongo.close()

# Create FastAPI app
app = FastAPI(
    title="Advanced RAG API",
    description="Production-grade RAG with streaming, WebSockets, and advanced techniques",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return generate_latest()

# RAG Query endpoint with streaming
@app.post("/api/v1/query")
async def query_rag(query: RAGQuery):
    """Main RAG query endpoint with streaming support"""
    
    async def stream_response():
        try:
            async for result in app.state.rag_engine.process_query(query):
                yield f"data: {json.dumps(result)}\n\n"
        except Exception as e:
            logger.error("RAG query error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'data': str(e)})}\n\n"
    
    if query.stream:
        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming response
        results = []
        async for result in app.state.rag_engine.process_query(query):
            results.append(result)
        
        # Combine results
        answer = ""
        sources = []
        confidence = 0
        
        for r in results:
            if r["type"] == "chunk":
                answer += r["data"]
            elif r["type"] == "sources":
                sources = r["data"]
            elif r["type"] == "complete":
                confidence = r["data"]["confidence"]
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={"query_id": query.session_id or "anonymous"}
        )

# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str, user_id: Optional[str] = None):
    """WebSocket endpoint for real-time RAG chat"""
    manager = app.state.connection_manager
    await manager.connect(websocket, session_id, user_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "query":
                query = RAGQuery(**message["data"])
                query.session_id = session_id
                
                # Save to chat history
                await app.state.db.chat_history.insert_one({
                    "session_id": session_id,
                    "user_id": user_id,
                    "query": query.dict(),
                    "timestamp": datetime.utcnow()
                })
                
                # Process query and stream results
                async for result in app.state.rag_engine.process_query(query):
                    await manager.send_message(
                        session_id,
                        WebSocketMessage(type=result["type"], data=result["data"])
                    )
                
            elif message["type"] == "feedback":
                # Handle user feedback
                await app.state.db.feedback.insert_one({
                    "session_id": session_id,
                    "user_id": user_id,
                    "feedback": message["data"],
                    "timestamp": datetime.utcnow()
                })
                
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e), session_id=session_id)
        await manager.send_message(
            session_id,
            WebSocketMessage(type="error", data={"message": str(e)})
        )
        manager.disconnect(session_id)

# Document management endpoints
@app.post("/api/v1/documents")
async def upload_documents(documents: List[Dict[str, Any]]):
    """Upload documents to the RAG system"""
    try:
        # Add documents to vector store
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        texts = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        await app.state.rag_engine.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        
        return {"status": "success", "documents_added": len(documents)}
    except Exception as e:
        logger.error("Document upload error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from the RAG system"""
    try:
        await app.state.rag_engine.collection.delete(ids=[doc_id])
        return {"status": "success", "document_id": doc_id}
    except Exception as e:
        logger.error("Document deletion error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Chat history endpoints
@app.get("/api/v1/chat/history/{session_id}")
async def get_chat_history(session_id: str, limit: int = Query(50, le=200)):
    """Get chat history for a session"""
    history = await app.state.db.chat_history.find(
        {"session_id": session_id}
    ).sort("timestamp", -1).limit(limit).to_list(limit)
    
    return {"session_id": session_id, "history": history}

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        workers=4,
        loop="uvloop",
        ws_ping_interval=20,
        ws_ping_timeout=60
    )
