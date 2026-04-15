import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
import uuid

class ChromaStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.DB_DIR)
        self.collection = self.client.get_or_create_collection(name="conversations")

    def add_interaction(self, user_text: str, agent_text: str, metadata: dict = None):
        """Store a turn of conversation into the memory."""
        doc_id = str(uuid.uuid4())
        content = f"User: {user_text}\nAgent: {agent_text}"
        
        meta = metadata or {}
        meta["user_query"] = user_text
        meta["timestamp"] = str(uuid.uuid1())
        
        self.collection.add(
            documents=[content],
            metadatas=[meta],
            ids=[doc_id]
        )

    def get_context(self, n_results: int = 5) -> str:
        """Retrieve recent interaction contexts if needed. For now just retrieving general docs."""
        try:
            results = self.collection.get(limit=n_results)
            if not results or not results['documents']:
                return ""
            return "\n---\n".join(results['documents'])
        except Exception:
            return ""

    def search_memory(self, query: str, n_results: int = 5):
        """Search memory semantically (Requires an embedding function mapped in local chroma, defaulting to all-MiniLM-L6-v2)"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if not results or not results['documents']:
            return []
        # Return flattened documents list
        return results['documents'][0]

chroma_store = ChromaStore()
