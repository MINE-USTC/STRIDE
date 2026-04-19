import gc
import os
from typing import List, Dict, Union
import numpy as np
import faiss
from tqdm import tqdm
import torch

class DenseRetriever:
    """Dense Retriever for efficient document search using various embedding models"""
    
    def __init__(self, model, tokenizer, batch_size=32, dim=768):
        self.index =faiss.IndexFlatIP(dim)
        # Maintaining the document data
        self.documents: List[str] = []
        self.titles: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        # self.model.get_embedding = lambda x: get_embed(model, tokenizer, x)
        self.ctr = 0
        
    # def get_embed(self, text):
    #     device = self.model.device
    #     tokens = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(device)
        
    #     with torch.no_grad():
    #         embeddings = self.model(**tokens).cpu().numpy()
    #     return embeddings
    def get_embed(self, texts):
        """
        Encode a single string or a list of strings into embeddings.
        
        Args:
            texts: str or List[str]
        
        Returns:
            np.ndarray of shape (N, dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        device = self.model.device
        tokens = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512  # 可选，防止过长
        ).to(device)
        
        with torch.no_grad():
            embeddings = self.model(**tokens)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)  # 推荐：L2 归一化（FAISS IP 需要）
            embeddings = embeddings.cpu().numpy()
        
        return embeddings

    def add_docs(self, document_texts, titles, embeds=None):
        if embeds is None:
            embeds = self.get_embed(document_texts)
        else:
            ### 如果是list, 转成 np array
            embeds = np.array(embeds)
        # print(document_texts)
        # print(embeds.shape)
        # print(embeds.dtype)
        
        self.index.add(embeds)
        self.documents.extend(document_texts)  # store the original document text
        self.titles.extend(titles)  # store the document titles
        self.embeddings.extend(embeds)  # store the document embedding
        self.ctr += len(document_texts)

    def add_doc(self, document_text, title, embed=None):
        if embed is None:
            embed = self.get_embed([document_text])
        self.add_docs([document_text], [title], [embed])
        
    def load_index(self, index_path: str = None):
        """Load the FAISS index and documents from disk"""
        try:
            # Load document data
            data = np.load(os.path.join(index_path, 'document.vecstore.npz'), allow_pickle=True)
            self.documents, self.embeddings = data['documents'].tolist(), data['embeddings'].tolist()
            self.titles = data['titles'].tolist()
            self.ctr = len(self.documents)
            print(f"Loaded {self.ctr} documents from {index_path}")

            # Load FAISS index
            self.index = faiss.read_index(os.path.join(index_path, 'faiss.index'))
            print(f"Index loaded successfully from {index_path}")

            # Cleanup
            del data
            gc.collect()

        except Exception as e:
            raise RuntimeError(f"Failed to load index from {index_path}: {str(e)}")

    def save_index(self, index_path: str = None):
        """Save the FAISS index and documents to disk"""
        if not self.index or not self.embeddings or not self.documents:
            raise ValueError("No index data to save")

        try:
            # Create directory if needed
            os.makedirs(index_path, exist_ok=True)
            print(f"Saving index to: {index_path}")

            # Save document data
            np.savez(
                os.path.join(index_path, 'document.vecstore'),
                embeddings=self.embeddings,
                documents=self.documents,
                titles=self.titles
            )

            # Save FAISS index
            faiss.write_index(self.index, os.path.join(index_path, 'faiss.index'))
            print(f"Index saved successfully to {index_path}, total documents: {self.ctr}")

        except Exception as e:
            raise RuntimeError(f"Failed to save index to {index_path}: {str(e)}")
        
    def build_from_texts(self, corpus: List[str]):
        """
        Index plain text passages in batches (empty title for each row).

        Prefer ``add_docs(texts, titles)`` when titles are available so retrieval
        matches the on-disk format used by ``supervisor`` / ``fallback_qa``.
        """
        if not corpus:
            return

        for i in tqdm(range(0, len(corpus), self.batch_size), desc="Building index"):
            batch = corpus[i : i + self.batch_size]
            titles = [""] * len(batch)
            self.add_docs(batch, titles)
            
    def clear(self):
        """Clear the index and all stored documents"""
        self.index.reset()
        self.documents = []
        self.titles = []
        self.embeddings = []
        self.ctr = 0
        print("Index and documents cleared")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Union[str, float]]]:
        """
        Retrieve the top_k documents relevant to the query.

        Args:
            query: Query string
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing retrieved documents and their scores
        """
        # Generate query embedding
        query_embedding = self.get_embed(query).astype('float32').reshape(1, -1)
        
        # print(query_embedding.shape)
        # print(query_embedding.dtype)
        # print(query_embedding)

        # Search index
        scores, indices = self.index.search(query_embedding, top_k)

        # Create results
        results = [
            {'text': self.documents[idx], 'title': self.titles[idx], 'score': score}
            for idx, score in zip(indices[0], scores[0])
        ]

        return results

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Dict[str, Union[str, float]]]]:
        """
        Retrieve top_k documents for each query in a batch.

        Args:
            queries: List of query strings
            top_k: Number of documents to retrieve per query

        Returns:
            List of lists, where each inner list contains top_k results for the corresponding query.
            Each result is a dict: {'text': ..., 'title': ..., 'score': ...}
        """
        if not queries:
            return []

        # Step 1: Encode all queries in batch
        query_embeddings = self.get_embed(queries)  # Shape: (N, dim)

        # Step 2: FAISS batch search
        scores, indices = self.index.search(query_embeddings, top_k)  # scores: (N, top_k), indices: (N, top_k)

        # Step 3: Assemble results per query
        all_results = []
        for i, query in enumerate(queries):
            results = []
            for j in range(len(indices[i])):
                idx = indices[i][j]
                if idx == -1:  # FAISS may return -1 if not enough docs
                    continue
                results.append({
                    'text': self.documents[idx],
                    'title': self.titles[idx],
                    'score': float(scores[i][j])
                })
            all_results.append(results)

        return all_results


# Index = DenseRetriever(contriever, contriever_tokenizer)