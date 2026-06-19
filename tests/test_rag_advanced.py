"""
tests/test_rag_advanced.py — Unit & Integration tests for RAPTOR-lite and Multi-Document Retrieval.
"""
from __future__ import annotations

import unittest
import numpy as np
from pathlib import Path

from backend.services.rag.raptor import gmm_cluster, RaptorIndexer
from backend.services.rag.retriever import HybridRetriever
from backend.services.rag.vector_store import VectorStoreManager


class TestRaptorClustering(unittest.TestCase):
    def test_gmm_cluster_basic(self):
        # 10 vectors of dimension 4
        # Create 2 distinct groups
        group_a = [np.array([1.0, 0.0, 0.0, 0.0]) + np.random.normal(0, 0.05, 4) for _ in range(5)]
        group_b = [np.array([0.0, 1.0, 0.0, 0.0]) + np.random.normal(0, 0.05, 4) for _ in range(5)]
        
        X = np.array(group_a + group_b, dtype=np.float32)
        
        # We expect 2 clusters
        clusters = gmm_cluster(X, k=2)
        
        self.assertEqual(len(clusters), 2)
        # Check that indices are clustered correctly
        cluster_1 = set(clusters[0])
        cluster_2 = set(clusters[1])
        
        # Verify indices from 0-4 are grouped together and 5-9 are grouped together
        first_group = set(range(5))
        second_group = set(range(5, 10))
        
        self.assertTrue(
            (cluster_1 == first_group and cluster_2 == second_group) or
            (cluster_1 == second_group and cluster_2 == first_group)
        )


class MockRepository:
    def __init__(self):
        self.saved_chunks = []
        self.saved_vectors = []
        self.model_name = None

    def save_chunks(self, chunks, vectors, model_name):
        self.saved_chunks.extend(chunks)
        self.saved_vectors.extend(vectors)
        self.model_name = model_name


class MockEmbeddingService:
    def embed_documents(self, texts, model):
        # Return dummy 4-dimensional vectors
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


class MockGenerator:
    pass


class TestRaptorIndexer(unittest.TestCase):
    def test_build_tree(self):
        repo = MockRepository()
        store = VectorStoreManager(Path("./storage/test_vector_store"))
        embedding = MockEmbeddingService()
        generator = MockGenerator()
        
        indexer = RaptorIndexer(repo, store, embedding, generator)
        
        base_chunks = [
            {"id": f"c_{i}", "document_id": "doc_123", "filename": "test.txt", "text": f"Đoạn văn số {i} nội dung mẫu."}
            for i in range(10)
        ]
        base_vectors = [[1.0 if j == i % 4 else 0.0 for j in range(4)] for i in range(10)]
        
        # Build RAPTOR tree
        indexer.build_tree("doc_123", base_chunks, base_vectors, "mock_embedding_model")
        
        # Check that summaries were generated and saved in the repository
        self.assertTrue(len(repo.saved_chunks) > 0)
        for chunk in repo.saved_chunks:
            self.assertEqual(chunk["document_id"], "doc_123")
            self.assertEqual(chunk["metadata"]["chunk_type"], "summary")
            self.assertEqual(chunk["metadata"]["level"], 1)
            self.assertTrue(len(chunk["metadata"]["child_chunk_ids"]) > 0)
            
        # Clean up vector store test artifacts
        store.delete_document("doc_123")


class TestMultiDocumentRetrieval(unittest.TestCase):
    def test_retrieval_filtering(self):
        store = VectorStoreManager(Path("./storage/test_vector_store"))
        
        # Insert chunks for doc A and doc B
        chunks_a = [
            {"id": "doc_a_c1", "document_id": "doc_a", "filename": "doc_a.txt", "text": "Học máy và trí tuệ nhân tạo.", "chunk_index": 0},
            {"id": "doc_a_c2", "document_id": "doc_a", "filename": "doc_a.txt", "text": "Học sâu ngày càng phát triển.", "chunk_index": 1}
        ]
        vectors_a = [[0.9, 0.1], [0.8, 0.2]]
        
        chunks_b = [
            {"id": "doc_b_c1", "document_id": "doc_b", "filename": "doc_b.txt", "text": "Báo cáo tài chính quý 1 năm nay.", "chunk_index": 0},
            {"id": "doc_b_c2", "document_id": "doc_b", "filename": "doc_b.txt", "text": "Tăng trưởng doanh thu khả quan.", "chunk_index": 1}
        ]
        vectors_b = [[0.1, 0.9], [0.2, 0.8]]
        
        store.upsert_chunks(chunks_a, vectors_a)
        store.upsert_chunks(chunks_b, vectors_b)
        
        # Query only doc A
        results_a = store.query(query_vector=[1.0, 0.0], top_k=5, document_ids=["doc_a"])
        self.assertTrue(all(r["document_id"] == "doc_a" for r in results_a))
        
        # Query only doc B
        results_b = store.query(query_vector=[0.0, 1.0], top_k=5, document_ids=["doc_b"])
        self.assertTrue(all(r["document_id"] == "doc_b" for r in results_b))
        
        # Query both
        results_both = store.query(query_vector=[0.5, 0.5], top_k=5, document_ids=["doc_a", "doc_b"])
        doc_ids_retrieved = {r["document_id"] for r in results_both}
        self.assertTrue(len(doc_ids_retrieved) > 1)
        
        # Clean up
        store.delete_document("doc_a")
        store.delete_document("doc_b")
