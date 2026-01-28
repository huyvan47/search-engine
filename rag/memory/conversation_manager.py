from rag.memory.memory_store import memory_store
from rag.memory.summarizer import summarize_to_fact
from rag.embedder import embed_text   # bạn đã có

EVENT_LOG = {}

def log_event(user_id, role, content):
    if user_id not in EVENT_LOG:
        EVENT_LOG[user_id] = []
    EVENT_LOG[user_id].append({
        "role": role,
        "content": content
    })

def build_conversation_text(user_id, max_turns=20):
    logs = EVENT_LOG.get(user_id, [])[-max_turns:]
    return "\n".join(f"{e['role']}: {e['content']}" for e in logs)

def write_memory(client, user_id, facts_json):
    for f in facts_json:
        emb = embed_text(client, f["fact"])
        memory_store.add(user_id, emb, f)

def read_memory(client, user_id, query, k=5):
    emb = embed_text(client, query)
    return memory_store.search(user_id, emb, k)
