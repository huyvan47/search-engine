import faiss
import numpy as np

class UserMemoryStore:
    def __init__(self, dim=1536):
        self.dim = dim
        self.user_indexes = {}   # user_id -> FAISS index
        self.user_facts = {}     # user_id -> list facts

    def _get_user_index(self, user_id):
        if user_id not in self.user_indexes:
            self.user_indexes[user_id] = faiss.IndexFlatIP(self.dim)
            self.user_facts[user_id] = []
        return self.user_indexes[user_id]

    def add(self, user_id, embedding, fact):
        index = self._get_user_index(user_id)
        index.add(np.array([embedding]).astype("float32"))
        self.user_facts[user_id].append(fact)

    def search(self, user_id, embedding, k=5):
        if user_id not in self.user_indexes:
            return []
        index = self.user_indexes[user_id]
        facts = self.user_facts[user_id]
        D, I = index.search(np.array([embedding]).astype("float32"), k)
        return [facts[i] for i in I[0] if i != -1]

memory_store = UserMemoryStore()
