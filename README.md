# 🚀 Advanced RAG System with FastAPI & WebSockets

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, featuring real-time streaming, WebSocket support, and cutting-edge RAG techniques.

## ✨ Features

### Advanced RAG Techniques
- **🔍 Hybrid Search**: Combines dense retrieval with semantic search
- **📊 Intelligent Reranking**: Cross-encoder models for optimal result ordering
- **🔄 Query Expansion**: Automatic synonym and semantic expansion
- **🎯 Self-RAG**: Confidence evaluation and adaptive retrieval
- **⚡ Long-Context RAG**: Handles extended document contexts efficiently

### Production Architecture
- **FastAPI**: High-performance async web framework
- **WebSockets**: Real-time bidirectional communication
- **Streaming**: Token-by-token response streaming (SSE & WebSocket)
- **Async/Await**: Non-blocking I/O throughout the application
- **Connection Management**: Robust WebSocket connection handling

### Scalability & Performance
- **Redis Caching**: Intelligent response caching with TTL
- **Connection Pooling**: Efficient database connections
- **Load Balancing Ready**: Multi-worker support with uvloop
- **Prometheus Metrics**: Built-in performance monitoring
- **Structured Logging**: Comprehensive logging with structlog

## 🛠️ Tech Stack

- **Framework**: FastAPI with uvicorn
- **Vector DB**: ChromaDB for embeddings
- **LLM**: OpenAI GPT-4 (configurable)
- **Embeddings**: Sentence Transformers (BAAI/bge-large)
- **Reranking**: Cross-encoder models
- **Caching**: Redis
- **Database**: MongoDB for chat history
- **Monitoring**: Prometheus + OpenTelemetry ready

## 📋 Prerequisites

- Python 3.8+
- Redis server
- MongoDB instance
- ChromaDB server
- OpenAI API key

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/advanced-rag-system.git
cd advanced-rag-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
```

### 3. Set environment variables
```bash
export OPENAI_API_KEY="your-api-key"
export CHROMA_HOST="localhost"
export MONGODB_URI="mongodb://localhost:27017"
export REDIS_URL="redis://localhost:6379"
```

### 4. Run the application
```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --loop uvloop
```

## 📚 API Documentation

### REST Endpoints

#### Query RAG System
```http
POST /api/v1/query
Content-Type: application/json

{
  "query": "What are the latest RAG techniques?",
  "user_id": "user123",
  "session_id": "session456",
  "top_k": 5,
  "temperature": 0.7,
  "stream": true
}
```

#### Upload Documents
```http
POST /api/v1/documents
Content-Type: application/json

[
  {
    "id": "doc1",
    "content": "Document content here...",
    "metadata": {
      "source": "research_paper.pdf",
      "date": "2024-01-01"
    }
  }
]
```

#### Get Chat History
```http
GET /api/v1/chat/history/{session_id}?limit=50
```

### WebSocket Interface

Connect to WebSocket endpoint:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/chat/session123');

// Send query
ws.send(JSON.stringify({
  type: 'query',
  data: {
    query: 'Your question here',
    stream: true,
    top_k: 5
  }
}));

// Receive streaming response
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch(message.type) {
    case 'chunk':
      // Handle streaming text chunk
      console.log(message.data);
      break;
    case 'sources':
      // Handle source documents
      console.log('Sources:', message.data);
      break;
    case 'complete':
      // Handle completion
      console.log('Complete:', message.data);
      break;
  }
};
```

## 🔧 Configuration

### Vector Store Configuration
```python
config = {
    "embedding_model": "BAAI/bge-large-en-v1.5",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "chroma_host": "localhost",
    "chroma_port": 8000
}
```

### Redis Cache Settings
- Default TTL: 300 seconds (5 minutes)
- Configurable per endpoint

### Model Parameters
- **Temperature**: 0.0 - 2.0 (default: 0.7)
- **Top-K**: 1 - 20 (default: 5)
- **Max tokens**: Configurable per request

## 📊 Monitoring

### Prometheus Metrics
- `rag_queries_total`: Total number of RAG queries
- `rag_query_duration_seconds`: Query processing duration
- `websocket_connections_total`: Total WebSocket connections

Access metrics at: `http://localhost:8000/metrics`

### Health Check
```http
GET /health
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 🐳 Docker Deployment

### Build image
```bash
docker build -t advanced-rag:latest .
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Docker Compose Configuration
```yaml
version: '3.8'

services:
  rag-api:
    image: advanced-rag:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - CHROMA_HOST=chroma
      - MONGODB_URI=mongodb://mongo:27017
      - REDIS_URL=redis://redis:6379
    depends_on:
      - chroma
      - mongo
      - redis

  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"

  mongo:
    image: mongo:latest
    ports:
      - "27017:27017"

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
```

## 🚀 Production Deployment

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: advanced-rag:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: openai-key
```

### AWS ECS
- Use Fargate for serverless deployment
- Configure ALB for load balancing
- Set up CloudWatch for logging

### Performance Tuning
- **Workers**: Set based on CPU cores (2 * CPU + 1)
- **Connection Pool**: Adjust based on concurrent users
- **Cache TTL**: Balance between freshness and performance
- **Chunk Size**: Optimize for streaming performance

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 🙏 Acknowledgments

- FastAPI for the amazing async framework
- Anthropic and OpenAI for LLM capabilities
- The open-source RAG community for continuous innovations


---

Built with ❤️ for the AI community
