"""
embedding.py: Modified version that only reuses embeddings at paper level (level 1)
"""

import os
import pickle
import time
import numpy as np
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

import openai

class OpenAIEmbeddingGenerator:
    def __init__(
        self,
        openai_key=None,
        model_name="text-embedding-ada-002",
        embeddings_path="embeddings",
        experiment_subfolder=None
    ):
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_key:
            raise ValueError("OpenAI API key not found. Please set env var OPENAI_API_KEY or pass openai_key.")
        openai.api_key = self.openai_key

        self.model_name = model_name
        self.base_embeddings_path = Path(embeddings_path)
        self.experiment_path = self.base_embeddings_path / experiment_subfolder if experiment_subfolder else self.base_embeddings_path

        self.base_embeddings_path.mkdir(parents=True, exist_ok=True)
        if experiment_subfolder:
            self.experiment_path.mkdir(parents=True, exist_ok=True)
        
    def _load_paper_embedding_cache(self) -> dict:
        """从顶层目录加载 paper embedding 缓存"""
        paper_embedding_path = self.base_embeddings_path / "paper_embedding.pkl"
        if paper_embedding_path.exists():
            with open(paper_embedding_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded paper embedding cache with {len(cache)} entries from {paper_embedding_path}")
            return cache
        else:
            print(f"No global paper embedding cache found at {paper_embedding_path}, returning empty dict.")
            return {}

    def _save_paper_embedding_cache(self, cache: dict):
        """保存到顶层目录"""
        paper_embedding_path = self.base_embeddings_path / "paper_embedding.pkl"
        with open(paper_embedding_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved paper embedding cache with {len(cache)} entries to {paper_embedding_path}")

    def get_embeddings(self, texts, cache_file: Path, is_paper: bool=False) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        
        global_cache = self._load_paper_embedding_cache() if is_paper else {}
        level_cache = {}
        
        if is_paper:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    level_cache = pickle.load(f)
                print(f"Loaded {len(level_cache)} embeddings from {cache_file}")
            else:
                print(f"No existing cache at {cache_file}; creating a new one.")

            for text_, emb_ in global_cache.items():
                if text_ not in level_cache:
                    level_cache[text_] = emb_

        texts_to_embed = texts if not is_paper else [t for t in texts if t not in level_cache]
        
        has_new_embeddings = len(texts_to_embed) > 0

        if texts_to_embed:
            print(f"Generating {len(texts_to_embed)} new embeddings via OpenAI({self.model_name})...")
            for idx, text in enumerate(texts_to_embed):
                try:
                    response = client.embeddings.create(
                        input=text,
                        model=self.model_name
                    )
                    emb = response.data[0].embedding
                    
                    if is_paper:
                        level_cache[text] = emb
                        global_cache[text] = emb
                    else:
                        if idx == 0:
                            all_embeddings = np.zeros((len(texts), len(emb)))
                        all_embeddings[idx] = emb
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache_file, all_embeddings)

                    if (idx+1) % 10 == 0 or (idx+1) == len(texts_to_embed):
                        if is_paper:
                            with open(cache_file, 'wb') as f:
                                pickle.dump(level_cache, f)
                            if has_new_embeddings:
                                self._save_paper_embedding_cache(global_cache)
                        print(f"Saved embeddings at {idx+1}/{len(texts_to_embed)}")

                    time.sleep(0.2)

                except Exception as e:
                    print(f"Error generating embedding for text: {e}")
                    vec = np.zeros(1536)
                    if is_paper:
                        level_cache[text] = vec
                        global_cache[text] = vec
                    else:
                        if idx == 0:
                            all_embeddings = np.zeros((len(texts), 1536))
                        all_embeddings[idx] = vec
        else:
            print("All embeddings for these texts are found in cache.")

        if is_paper:
            with open(cache_file, 'wb') as f:
                pickle.dump(level_cache, f)
            if has_new_embeddings:
                self._save_paper_embedding_cache(global_cache)
            all_embeddings = np.array([level_cache[t] for t in texts])

        return all_embeddings

class QwenEmbeddingGenerator:
    def __init__(
        self,
        model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
        embeddings_path="embeddings",
        experiment_subfolder=None,
        device=None
    ):
        """
        Initialize the embedding generator for using Qwen embeddings with caching
        
        Args:
            model_name: Qwen embedding model name
            embeddings_path: Directory to save embeddings
            experiment_subfolder: Optional subfolder for this experiment
            device: Device to run model on ('cpu', 'cuda', etc.) or None for auto-detection
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        
        self.model_name = model_name
        
        self.embedding_dim = 3584
        
        self.base_embeddings_path = Path(embeddings_path)
        self.experiment_path = self.base_embeddings_path / experiment_subfolder if experiment_subfolder else self.base_embeddings_path

        self.base_embeddings_path.mkdir(parents=True, exist_ok=True)
        if experiment_subfolder:
            self.experiment_path.mkdir(parents=True, exist_ok=True)
        
    def _load_paper_embedding_cache(self) -> dict:
        """Load paper embedding cache from top-level directory"""
        paper_embedding_path = self.base_embeddings_path / "paper_embedding_qwen.pkl"
        if paper_embedding_path.exists():
            with open(paper_embedding_path, 'rb') as f:
                cache = pickle.load(f)
            print(f"Loaded paper embedding cache with {len(cache)} entries from {paper_embedding_path}")
            return cache
        else:
            print(f"No global paper embedding cache found at {paper_embedding_path}, returning empty dict.")
            return {}

    def _save_paper_embedding_cache(self, cache: dict):
        """Save to top-level directory"""
        paper_embedding_path = self.base_embeddings_path / "paper_embedding_qwen.pkl"
        with open(paper_embedding_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f"Saved paper embedding cache with {len(cache)} entries to {paper_embedding_path}")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text string using Qwen model"""
        if not text or text.strip() == "":
            return np.zeros(self.embedding_dim)
        
        text = text.strip().replace("\n", " ")
        
        try:
            prompt_text = f"Instruct: Extract key information from academic paper\nQuery: {text}"
            embedding = self.model.encode(prompt_text, prompt_name="query", convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return np.zeros(self.embedding_dim)

    def get_embeddings(self, texts, cache_file: Path, is_paper: bool=False) -> np.ndarray:
        """
        Get embeddings for multiple texts with caching
        
        Args:
            texts: List of text strings to embed
            cache_file: Path to the cache file
            is_paper: Whether these are paper-level texts (for caching purposes)
            
        Returns:
            numpy array of embeddings
        """
        global_cache = self._load_paper_embedding_cache() if is_paper else {}
        level_cache = {}
        
        if is_paper:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    level_cache = pickle.load(f)
                print(f"Loaded {len(level_cache)} embeddings from {cache_file}")
            else:
                print(f"No existing cache at {cache_file}; creating a new one.")

            for text_, emb_ in global_cache.items():
                if text_ not in level_cache:
                    level_cache[text_] = emb_

        texts_to_embed = texts if not is_paper else [t for t in texts if t not in level_cache]
        
        has_new_embeddings = len(texts_to_embed) > 0

        if texts_to_embed:
            print(f"Generating {len(texts_to_embed)} new embeddings via Qwen({self.model_name})...")
            for idx, text in enumerate(texts_to_embed):
                try:
                    emb = self.get_embedding(text)
                    
                    if is_paper:
                        level_cache[text] = emb
                        global_cache[text] = emb
                    else:
                        if idx == 0:
                            all_embeddings = np.zeros((len(texts), self.embedding_dim))
                        all_embeddings[idx] = emb
                        cache_file.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache_file, all_embeddings)

                    if (idx+1) % 10 == 0 or (idx+1) == len(texts_to_embed):
                        if is_paper:
                            with open(cache_file, 'wb') as f:
                                pickle.dump(level_cache, f)
                            if has_new_embeddings:
                                self._save_paper_embedding_cache(global_cache)
                        print(f"Saved embeddings at {idx+1}/{len(texts_to_embed)}")

                    time.sleep(0.05)

                except Exception as e:
                    print(f"Error generating embedding for text: {e}")
                    vec = np.zeros(self.embedding_dim)
                    if is_paper:
                        level_cache[text] = vec
                        global_cache[text] = vec
                    else:
                        if idx == 0:
                            all_embeddings = np.zeros((len(texts), self.embedding_dim))
                        all_embeddings[idx] = vec
        else:
            print("All embeddings for these texts are found in cache.")

        if is_paper:
            with open(cache_file, 'wb') as f:
                pickle.dump(level_cache, f)
            if has_new_embeddings:
                self._save_paper_embedding_cache(global_cache)
            all_embeddings = np.array([level_cache[t] for t in texts])

        return all_embeddings