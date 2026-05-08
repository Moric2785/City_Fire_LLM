# FLLM/src/rag/retriever.py
try:
    import faiss
except ImportError:
    print("Warning: faiss not installed, please run: pip install faiss-cpu")
    faiss = None
import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
import hashlib
import pickle
import os
from functools import lru_cache

class Retriever:
    """
    Class for retrieving relevant information from knowledge graphs using embedding vectors.
    Supports caching, batch processing, and performance optimization.
    """
    def __init__(self, node_embeddings: torch.Tensor, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', 
                 cache_size: int = 1000, enable_cache: bool = True):
        """
        Initialize Retriever.

        Args:
            node_embeddings (torch.Tensor): Embedding vectors for knowledge graph nodes.
            model_name (str): Pre-trained model name for encoding queries.
            cache_size (int): Cache size.
            enable_cache (bool): Whether to enable caching.
        """
        self.logger = logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        
        # Initialize cache
        if self.enable_cache:
            self.query_cache = {}
            self.embedding_cache = {}
        
        if faiss is None:
            raise ImportError("faiss not installed, please run: pip install faiss-cpu")
            
        # 1. Prepare FAISS index
        embedding_dim = node_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(embedding_dim)  # Use inner product as similarity
        
        # Normalize embedding vectors for inner product search
        node_embeddings_np = node_embeddings.numpy()
        faiss.normalize_L2(node_embeddings_np)
        self.index.add(node_embeddings_np)
        
        # Ensure query embedding dimension matches index
        self.embedding_dim = embedding_dim
        self.logger.info(f"FAISS index created with {self.index.ntotal} vectors.")

        # 2. Load query encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.logger.info(f"Query encoder '{model_name}' loaded to {self.device}.")
        
        # 3. Performance statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_query_time': 0.0
        }

    def _get_query_hash(self, query: str) -> str:
        """
        Generate hash value for query for caching.
        """
        return hashlib.md5(query.encode('utf-8')).hexdigest()

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode text query to vector with caching support.
        """
        if self.enable_cache:
            query_hash = self._get_query_hash(query)
            if query_hash in self.embedding_cache:
                self.stats['cache_hits'] += 1
                return self.embedding_cache[query_hash]
            else:
                self.stats['cache_misses'] += 1
        
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Average pooling
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Ensure query embedding dimension matches index
        if query_embedding.shape[1] != self.embedding_dim:
            # If dimensions don't match, use linear transformation or truncate/pad
            if query_embedding.shape[1] > self.embedding_dim:
                query_embedding = query_embedding[:, :self.embedding_dim]
            else:
                # Pad to target dimension
                padding = np.zeros((query_embedding.shape[0], self.embedding_dim - query_embedding.shape[1]))
                query_embedding = np.concatenate([query_embedding, padding], axis=1)
        
        faiss.normalize_L2(query_embedding)
        
        # Cache embedding vector
        if self.enable_cache:
            if len(self.embedding_cache) >= self.cache_size:
                # Simple LRU strategy: remove oldest cache item
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            self.embedding_cache[query_hash] = query_embedding
        
        return query_embedding

    def retrieve(self, query: str, k: int = 5) -> List[int]:
        """
        检索与查询最相关的前k个节点。

        Args:
            query (str): 输入的文本查询 (火灾事件prompt)。
            k (int): 要检索的节点数量。

        Returns:
            List[int]: 前k个相关节点的索引列表。
        """
        import time
        start_time = time.time()
        
        # 检查缓存
        if self.enable_cache:
            query_hash = self._get_query_hash(f"{query}_{k}")
            if query_hash in self.query_cache:
                self.stats['cache_hits'] += 1
                return self.query_cache[query_hash]
            else:
                self.stats['cache_misses'] += 1
        
        query_embedding = self._encode_query(query)
        _, indices = self.index.search(query_embedding, k)
        result = indices.flatten().tolist()
        
        # 缓存结果
        if self.enable_cache:
            if len(self.query_cache) >= self.cache_size:
                # 简单的LRU策略
                oldest_key = next(iter(self.query_cache))
                del self.query_cache[oldest_key]
            self.query_cache[query_hash] = result
        
        # 更新统计信息
        query_time = time.time() - start_time
        self.stats['total_queries'] += 1
        self.stats['avg_query_time'] = (
            (self.stats['avg_query_time'] * (self.stats['total_queries'] - 1) + query_time) 
            / self.stats['total_queries']
        )
        
        return result

    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[int]]:
        """
        批量检索，提高效率。

        Args:
            queries (List[str]): 查询列表。
            k (int): 每个查询要检索的节点数量。

        Returns:
            List[List[int]]: 每个查询的检索结果列表。
        """
        results = []
        for query in queries:
            result = self.retrieve(query, k=k)
            results.append(result)
        return results

    def retrieve_with_scores(self, query: str, k: int = 5) -> Tuple[List[int], List[float]]:
        """
        检索并返回相似度分数。

        Args:
            query (str): 输入的文本查询。
            k (int): 要检索的节点数量。

        Returns:
            Tuple[List[int], List[float]]: 节点索引列表和相似度分数列表。
        """
        query_embedding = self._encode_query(query)
        scores, indices = self.index.search(query_embedding, k)
        return indices.flatten().tolist(), scores.flatten().tolist()

    def get_retrieval_stats(self) -> Dict:
        """
        获取检索统计信息。

        Returns:
            Dict: 统计信息字典。
        """
        stats = self.stats.copy()
        if self.enable_cache:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses']) if (stats['cache_hits'] + stats['cache_misses']) > 0 else 0
            stats['cache_size'] = len(self.query_cache)
            stats['embedding_cache_size'] = len(self.embedding_cache)
        return stats

    def clear_cache(self):
        """
        清空缓存。
        """
        if self.enable_cache:
            self.query_cache.clear()
            self.embedding_cache.clear()
            self.logger.info("缓存已清空")

    def save_retriever(self, save_path: str):
        """
        保存检索器。

        Args:
            save_path (str): 保存路径。
        """
        os.makedirs(save_path, exist_ok=True)
        
        # 保存FAISS索引
        faiss.write_index(self.index, os.path.join(save_path, 'faiss_index.bin'))
        
        # 保存模型配置
        config = {
            'model_name': self.model.config.name_or_path,
            'cache_size': self.cache_size,
            'enable_cache': self.enable_cache,
            'device': str(self.device)
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            import json
            json.dump(config, f, indent=2)
        
        # 保存缓存（如果启用）
        if self.enable_cache:
            cache_data = {
                'query_cache': self.query_cache,
                'embedding_cache': self.embedding_cache
            }
            with open(os.path.join(save_path, 'cache.pkl'), 'wb') as f:
                pickle.dump(cache_data, f)
        
        self.logger.info(f"检索器已保存到: {save_path}")

    def load_retriever(self, load_path: str):
        """
        加载检索器。

        Args:
            load_path (str): 加载路径。
        """
        # 加载FAISS索引
        if os.path.exists(os.path.join(load_path, 'faiss_index.bin')):
            self.index = faiss.read_index(os.path.join(load_path, 'faiss_index.bin'))
        
        # 加载配置
        config_path = os.path.join(load_path, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config = json.load(f)
                self.cache_size = config.get('cache_size', 1000)
                self.enable_cache = config.get('enable_cache', True)
        
        # 加载缓存
        cache_path = os.path.join(load_path, 'cache.pkl')
        if os.path.exists(cache_path) and self.enable_cache:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.query_cache = cache_data.get('query_cache', {})
                self.embedding_cache = cache_data.get('embedding_cache', {})
        
        self.logger.info(f"检索器已从 {load_path} 加载")
