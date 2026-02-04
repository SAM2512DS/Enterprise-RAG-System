# enterprise_rag_frontend_complete.py

import streamlit as st
import requests
import time
from datetime import datetime
from typing import List, Dict, Any

# ----------------------------
# Configuration
# ----------------------------
API_BASE = "http://localhost:8000"  # Change if backend runs elsewhere
HEADERS = {}

# ----------------------------
# Page Setup
# ----------------------------
st.set_page_config(
    page_title="Enterprise RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if "query_results" not in st.session_state:
    st.session_state.query_results = []
if "current_session" not in st.session_state:
    st.session_state.current_session = f"session_{int(time.time())}"

# ----------------------------
# Helper Functions
# ----------------------------
def update_headers():
    """Update headers with API key if provided"""
    if st.session_state.api_key:
        return {"X-API-Key": st.session_state.api_key}
    return {}

def api_call(method: str, endpoint: str, **kwargs):
    """Make API call with error handling"""
    try:
        headers = update_headers()
        if "headers" in kwargs:
            headers.update(kwargs["headers"])
        kwargs["headers"] = headers
        
        resp = requests.request(method, f"{API_BASE}{endpoint}", **kwargs)
        
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 403:
            st.error("❌ Invalid API Key")
            return None
        elif resp.status_code == 429:
            st.error("⏳ Rate limit exceeded. Please wait.")
            return None
        else:
            st.error(f"❌ API Error {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("🔌 Cannot connect to backend. Please ensure the server is running.")
        return None
    except Exception as e:
        st.error(f"⚠️ Unexpected error: {str(e)}")
        return None

def display_answer_with_sources(answer: str, sources: List[str]):
    """Display answer with expandable sources"""
    st.markdown("### 📝 Answer")
    st.markdown(answer)
    
    if sources and len(sources) > 0:
        st.markdown("### 📚 Sources")
        for i, src in enumerate(sources):
            with st.expander(f"Source {i+1} ({len(src)} chars)"):
                st.text(src)

def save_query_result(query: str, answer: str, cached: bool = False):
    """Save query result to session state"""
    result = {
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "query": query,
        "answer": answer[:500] + "..." if len(answer) > 500 else answer,
        "cached": cached
    }
    st.session_state.query_results.insert(0, result)
    if len(st.session_state.query_results) > 10:
        st.session_state.query_results.pop()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key
    st.subheader("🔑 Authentication")
    api_key = st.text_input(
        "API Key",
        type="password",
        value=st.session_state.api_key,
        help="Leave empty if backend has no authentication"
    )
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        HEADERS = update_headers()
        if api_key:
            st.success("✅ API Key updated")
    
    # Backend Status
    st.subheader("📡 Backend Status")
    if st.button("Check Connection"):
        result = api_call("GET", "/health")
        if result:
            st.success("✅ Backend connected")
            with st.expander("View Details"):
                st.json(result)
        else:
            st.error("❌ Backend offline")
    
    # Session Management
    st.subheader("💬 Chat Session")
    session_id = st.text_input(
        "Session ID",
        value=st.session_state.current_session,
        help="Unique ID for chat history"
    )
    if session_id != st.session_state.current_session:
        st.session_state.current_session = session_id
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New Session"):
            new_id = f"session_{int(time.time())}"
            st.session_state.current_session = new_id
            st.rerun()
    with col2:
        if st.button("🗑️ Clear Chat"):
            result = api_call("DELETE", f"/chat/{st.session_state.current_session}")
            if result:
                st.success("Chat history cleared")
                if st.session_state.current_session in st.session_state.chat_history:
                    del st.session_state.chat_history[st.session_state.current_session]
    
    # System Metrics
    st.subheader("📊 System Metrics")
    if st.button("Get Metrics"):
        result = api_call("GET", "/metrics")
        if result:
            with st.expander("View Metrics"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Requests", result.get("total_requests", 0))
                    st.metric("Total Queries", result.get("total_queries", 0))
                    st.metric("Cache Hits", result.get("cache_hits", 0))
                with col2:
                    st.metric("Total Documents", result.get("total_documents", 0))
                    st.metric("Total Chunks", result.get("total_chunks", 0))
                    st.metric("Uptime", f"{result.get('uptime_seconds', 0)//60:.0f} min")
    
    # Cache Management
    st.subheader("🧹 Cache Management")
    if st.button("Clear Cache"):
        result = api_call("POST", "/cache/clear")
        if result:
            st.success("✅ Cache cleared")
    
    # Configuration
    st.subheader("⚙️ System Config")
    if st.button("View Configuration"):
        result = api_call("GET", "/config")
        if result:
            with st.expander("Configuration"):
                st.json(result)

# ----------------------------
# Main Content
# ----------------------------
st.title("📚 Enterprise RAG System")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Upload", "🔍 Query", "💬 Chat", "📄 Documents", "📊 Evaluate", "📈 Analytics"
])

# ----------------------------
# Tab 1: Upload Documents
# ----------------------------
with tab1:
    st.header("📤 Upload Documents")
    
    # Single File Upload
    st.subheader("Single File Upload")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt", "csv", "json", "md"],
        key="single_upload"
    )
    
    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Selected: **{uploaded_file.name}** ({uploaded_file.size:,} bytes)")
        with col2:
            if st.button("Upload File", type="primary"):
                with st.spinner("Uploading..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                    result = api_call("POST", "/upload-file", files=files)
                    if result:
                        st.success(f"✅ Uploaded successfully!")
                        st.json(result)
    
    st.markdown("---")
    
    # Batch Upload
    st.subheader("Batch Upload")
    batch_files = st.file_uploader(
        "Upload multiple files",
        type=["pdf", "docx", "txt", "csv", "json", "md"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if batch_files:
        st.info(f"Selected {len(batch_files)} files")
        file_list = "\n".join([f"- {f.name} ({f.size:,} bytes)" for f in batch_files])
        st.markdown(file_list)
        
        if st.button("Upload Batch", type="primary"):
            with st.spinner("Uploading batch..."):
                files = []
                for f in batch_files:
                    files.append(("files", (f.name, f.getvalue())))
                
                result = api_call("POST", "/upload-batch", files=files)
                if result:
                    successful = sum(1 for r in result.get("results", []) if r.get("success"))
                    st.success(f"✅ Batch upload complete: {successful}/{len(batch_files)} successful")
                    with st.expander("View Details"):
                        st.json(result)

# ----------------------------
# Tab 2: Query
# ----------------------------
with tab2:
    st.header("🔍 Query Knowledge Base")
    
    # Query Type Selection
    col1, col2 = st.columns([2, 1])
    with col1:
        query_type = st.radio(
            "Query Type",
            ["Simple RAG", "Agentic RAG", "Streaming"],
            horizontal=True
        )
    with col2:
        k = st.number_input("Results (k)", min_value=1, max_value=20, value=5)
    
    # Advanced Options
    with st.expander("⚙️ Advanced Options"):
        col1, col2, col3 = st.columns(3)
        with col1:
            use_cache = st.checkbox("Use Cache", value=True)
        with col2:
            use_reranking = st.checkbox("Use Reranking", value=True)
        with col3:
            optimize_query = st.checkbox("Optimize Query", value=True)
    
    # Query Input
    query_input = st.text_area(
        "Enter your question",
        height=120,
        placeholder="Type your question here...",
        key="query_input"
    )
    
    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button("🚀 Run Query", type="primary", disabled=not query_input.strip()):
            if not query_input.strip():
                st.warning("Please enter a query")
            else:
                # Prepare payload
                payload = {
                    "query": query_input,
                    "k": k,
                    "use_cache": use_cache,
                    "use_reranking": use_reranking
                }
                
                # Select endpoint
                if query_type == "Simple RAG":
                    endpoint = "/query/simple"
                elif query_type == "Agentic RAG":
                    endpoint = "/query/agentic"
                else:  # Streaming
                    endpoint = "/query/stream"
                
                # Execute query
                if query_type == "Streaming":
                    with st.spinner("Streaming response..."):
                        answer_container = st.empty()
                        full_answer = ""
                        
                        try:
                            headers = update_headers()
                            resp = requests.post(
                                f"{API_BASE}{endpoint}",
                                json=payload,
                                headers=headers,
                                stream=True
                            )
                            
                            if resp.status_code == 200:
                                for chunk in resp.iter_content(chunk_size=512):
                                    if chunk:
                                        text = chunk.decode()
                                        full_answer += text
                                        answer_container.markdown(full_answer)
                                save_query_result(query_input, full_answer, False)
                                st.success("✅ Streaming complete!")
                            else:
                                st.error(f"Streaming failed: {resp.status_code}")
                        except Exception as e:
                            st.error(f"Streaming error: {str(e)}")
                else:
                    with st.spinner("Processing query..."):
                        result = api_call("POST", endpoint, json=payload)
                        if result:
                            display_answer_with_sources(
                                result.get("answer", ""),
                                result.get("sources", [])
                            )
                            
                            # Show additional info
                            with st.expander("📋 Query Details"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Sources", result.get("num_sources", 0))
                                with col2:
                                    cached = "✅" if result.get("cached", False) else "❌"
                                    st.metric("Cached", cached)
                                with col3:
                                    st.metric("Time", result.get("query_time", "N/A"))
                            
                            # For agentic RAG, show steps
                            if query_type == "Agentic RAG" and "steps" in result:
                                st.subheader("🤖 Agent Steps")
                                for i, step in enumerate(result["steps"]):
                                    st.write(f"{i+1}. {step}")
                            
                            save_query_result(query_input, result.get("answer", ""), result.get("cached", False))
    
    with col2:
        if st.button("🗑️ Clear"):
            query_input = ""
            st.rerun()
    
    # Recent Queries
    if st.session_state.query_results:
        st.markdown("---")
        st.subheader("📝 Recent Queries")
        for i, result in enumerate(st.session_state.query_results[:5]):
            with st.expander(f"{result['timestamp']} - {result['query'][:50]}..."):
                st.markdown(f"**Query:** {result['query']}")
                st.markdown(f"**Answer:** {result['answer']}")
                st.caption(f"Cached: {'✅' if result['cached'] else '❌'}")

# ----------------------------
# Tab 3: Chat
# ----------------------------
with tab3:
    st.header("💬 Multi-turn Chat")
    
    # Session Info
    st.info(f"Session ID: `{st.session_state.current_session}`")
    
    # Initialize chat history for this session
    if st.session_state.current_session not in st.session_state.chat_history:
        st.session_state.chat_history[st.session_state.current_session] = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history[st.session_state.current_session]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to history and display
        st.session_state.chat_history[st.session_state.current_session].append({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "session_id": st.session_state.current_session,
                    "query": prompt
                }
                result = api_call("POST", "/chat", json=payload)
                
                if result:
                    answer = result.get("answer", "")
                    st.markdown(answer)
                    
                    # Add assistant message to history
                    st.session_state.chat_history[st.session_state.current_session].append({
                        "role": "assistant",
                        "content": answer,
                        "timestamp": datetime.now().strftime("%H:%M:%S")
                    })
                else:
                    st.error("Failed to get response")

# ----------------------------
# Tab 4: Documents
# ----------------------------
with tab4:
    st.header("📄 Document Management")
    
    # Get documents
    result = api_call("GET", "/documents")
    
    if result:
        documents = result.get("documents", [])
        total_docs = result.get("total", 0)
        
        st.metric("Total Documents", total_docs)
        
        if documents:
            # Summary stats
            col1, col2, col3 = st.columns(3)
            total_chunks = sum(d.get("chunks", 0) for d in documents)
            total_size = sum(d.get("size", 0) for d in documents)
            
            with col1:
                st.metric("Total Chunks", total_chunks)
            with col2:
                st.metric("Total Size", f"{total_size:,} bytes")
            with col3:
                avg_chunks = total_chunks / len(documents) if documents else 0
                st.metric("Avg Chunks/Doc", f"{avg_chunks:.1f}")
            
            st.markdown("---")
            
            # Document list with delete functionality
            st.subheader("Document List")
            for doc in documents:
                with st.expander(f"📄 {doc['filename']}"):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**Extension:** {doc.get('extension', 'N/A')}")
                        st.write(f"**Size:** {doc.get('size', 0):,} bytes")
                        st.write(f"**Chunks:** {doc.get('chunks', 0)}")
                        st.write(f"**Uploaded:** {doc.get('uploaded_at', 'N/A')[:19]}")
                    with col2:
                        if st.button("🗑️ Delete", key=f"del_{doc['filename']}"):
                            if api_call("DELETE", f"/documents/{doc['filename']}"):
                                st.success(f"Deleted {doc['filename']}")
                                st.rerun()
        else:
            st.info("No documents uploaded yet.")
    else:
        st.error("Failed to fetch documents")

# ----------------------------
# Tab 5: Evaluate
# ----------------------------
with tab5:
    st.header("📊 RAG Evaluation")
    
    st.markdown("Evaluate the quality of RAG responses")
    
    eval_query = st.text_area(
        "Query to evaluate",
        height=100,
        placeholder="Enter a query to evaluate the RAG system's performance..."
    )
    
    if st.button("🔍 Evaluate Query", type="primary", disabled=not eval_query.strip()):
        with st.spinner("Evaluating..."):
            payload = {"query": eval_query}
            result = api_call("POST", "/evaluate", json=payload)
            
            if result:
                st.success("✅ Evaluation complete!")
                
                # Display answer
                st.subheader("🤖 Generated Answer")
                st.markdown(result.get("answer", ""))
                
                # Display evaluation scores
                evaluation = result.get("evaluation", {})
                if evaluation and "error" not in evaluation:
                    st.subheader("📈 Evaluation Scores")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        score = evaluation.get("context_relevance", 0)
                        st.metric("Context Relevance", f"{score}/5")
                        st.progress(score / 5)
                    
                    with col2:
                        score = evaluation.get("faithfulness", 0)
                        st.metric("Faithfulness", f"{score}/5")
                        st.progress(score / 5)
                    
                    with col3:
                        score = evaluation.get("answer_completeness", 0)
                        st.metric("Completeness", f"{score}/5")
                        st.progress(score / 5)
                    
                    with col4:
                        score = evaluation.get("overall_score", 0)
                        st.metric("Overall Score", f"{score:.1f}/5")
                        st.progress(score / 5)
                    
                    # Feedback
                    if "feedback" in evaluation:
                        st.info(f"📝 Feedback: {evaluation['feedback']}")
                
                # Display sources
                st.subheader("📚 Sources Used")
                sources = result.get("sources", [])
                for i, src in enumerate(sources):
                    with st.expander(f"Source {i+1}"):
                        st.text(src[:800] + "..." if len(src) > 800 else src)
            else:
                st.error("Evaluation failed")

# ----------------------------
# Tab 6: Analytics
# ----------------------------
with tab6:
    st.header("📈 System Analytics")
    
    # Get metrics
    result = api_call("GET", "/metrics")
    
    if result:
        # Key metrics
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Requests", result.get("total_requests", 0))
        with col2:
            st.metric("Total Queries", result.get("total_queries", 0))
        with col3:
            st.metric("Total Uploads", result.get("total_uploads", 0))
        with col4:
            uptime = result.get("uptime_seconds", 0)
            hours = uptime // 3600
            minutes = (uptime % 3600) // 60
            st.metric("Uptime", f"{hours}h {minutes}m")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cache_hits = result.get("cache_hits", 0)
            cache_misses = result.get("cache_misses", 0)
            total = cache_hits + cache_misses
            hit_rate = (cache_hits / total * 100) if total > 0 else 0
            st.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        
        with col2:
            avg_query_time = result.get("avg_query_time", 0)
            st.metric("Avg Query Time", f"{avg_query_time:.2f}s")
        
        with col3:
            avg_upload_time = result.get("avg_upload_time", 0)
            st.metric("Avg Upload Time", f"{avg_upload_time:.2f}s")
        
        # Error rate
        st.markdown("---")
        st.subheader("⚠️ Error Analysis")
        
        errors = result.get("errors", 0)
        requests = result.get("total_requests", 1)
        error_rate = (errors / requests * 100) if requests > 0 else 0
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.progress(1 - (error_rate / 100))
        with col2:
            st.metric("Error Rate", f"{error_rate:.2f}%")
        
        # Raw data
        st.markdown("---")
        with st.expander("📋 View Raw Metrics Data"):
            st.json(result)
    else:
        st.error("Failed to load metrics")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Enterprise RAG System v2.0 | Complete Implementation
    </div>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Run Instructions
# ----------------------------
# To run:
# 1. Ensure backend is running: python enterprise_rag_backend.py
# 2. Run this frontend: streamlit run enterprise_rag_frontend_complete.py
# 3. Open browser to http://localhost:8501