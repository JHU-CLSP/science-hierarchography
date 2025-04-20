import json
import os
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


class QwenKeyEmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-Qwen2-7B-instruct",
        embeddings_path: str = "./embeddings",
        device: str = None
    ):
        """
        Initialize the embedding generator for individual paper keys using Qwen
        
        Args:
            model_name: Qwen embedding model name
            embeddings_path: Directory to save embeddings
            device: Device to run model on ('cpu', 'cuda', etc.) or None for auto-detection
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=self.device)
        
        self.model_name = model_name
        
        self.embedding_dim = 3584
        
        self.embeddings_path = Path(embeddings_path)
        self.embeddings_path.mkdir(parents=True, exist_ok=True)
        
        self.backup_files = []
        
        self.embed_keys = [
            ("problem", "overarching problem domain"),
            ("problem", "challenges/difficulties"),
            ("problem", "research question/goal"),
            ("problem", "novelty of the problem"),
            ("problem", "knowns or prior work"),
            ("solution", "overarching solution domain"),
            ("solution", "knowns or prior work"),
            ("solution", "solution approach"),
            ("solution", "novelty of the solution"),
            ("results", "findings/results"),
            ("results", "potential impact of the results")
        ]
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text string using Qwen model"""
        if not text or text.strip() == "":
            return [0.0] * self.embedding_dim
        
        text = text.strip().replace("\n", " ")
        
        try:
            prompt_text = f"Instruct: Extract key information from academic paper\nQuery: {text}"
            embedding = self.model.encode(prompt_text, prompt_name="query", convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * self.embedding_dim
    
    def load_papers(self, input_folder: Path) -> List[Dict]:
        """Load all papers from the input folder"""
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        files = list(input_folder.glob("*.json"))
        if not files:
            raise ValueError(f"No JSON files found in {input_folder}")
        
        all_papers = []
        for file_path in tqdm(files, desc="Reading papers"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    data["file_name"] = file_path.name
                    all_papers.append(data)
                elif isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict):
                            item["file_name"] = file_path.name
                            all_papers.append(item)
            except Exception as e:
                print(f"[WARNING] Error processing {file_path}: {e}")
        
        print(f"Loaded {len(all_papers)} papers")
        return all_papers
    
    def cleanup_backups(self):
        """Remove all backup files created during processing"""
        if not self.backup_files:
            return
        
        print(f"Cleaning up {len(self.backup_files)} backup files...")
        for backup_file in self.backup_files:
            try:
                if backup_file.exists():
                    backup_file.unlink()
                    print(f"Deleted backup: {backup_file.name}")
            except Exception as e:
                print(f"Warning: Could not delete {backup_file}: {e}")
    
    def generate_key_embeddings(self, input_folder: str, output_file: str, keep_backups: bool = False):
        """
        Generate embeddings for each key in all papers
        
        Args:
            input_folder: Folder containing paper JSON files
            output_file: Path to save the embeddings pickle file
            keep_backups: If True, don't delete backup files when done
        """
        input_path = Path(input_folder)
        papers = self.load_papers(input_path)
        
        paper_embeddings = {}
        
        self.backup_files = []
        
        for paper_idx, paper in enumerate(tqdm(papers, desc="Processing papers")):
            paper_title = paper.get("title", f"paper_{paper.get('file_name', f'unknown_{paper_idx}')}")
            
            paper_embeddings[paper_title] = {
                "file_name": paper.get("file_name", ""),
                "key_embeddings": {}
            }
            
            for category, key in self.embed_keys:
                key_path = f"{category}.{key}"
                
                try:
                    text = paper.get(category, {}).get(key, "")
                    if not text:
                        text = ""
                    
                    embedding = self.get_embedding(text)
                    
                    paper_embeddings[paper_title]["key_embeddings"][key_path] = {
                        "text": text,
                        "embedding": embedding
                    }
                except Exception as e:
                    print(f"Error processing key {key_path} for paper {paper_title}: {e}")
                    paper_embeddings[paper_title]["key_embeddings"][key_path] = {
                        "text": "",
                        "embedding": [0.0] * self.embedding_dim
                    }
            
            if (paper_idx + 1) % 10 == 0:
                backup_file = self.embeddings_path / f"key_embeddings_backup_{paper_idx + 1}.pkl"
                with open(backup_file, 'wb') as f:
                    pickle.dump(paper_embeddings, f)
                print(f"Saved backup embeddings after {paper_idx + 1} papers to {backup_file}")
                self.backup_files.append(backup_file)
        
        output_path = Path(output_file)
        with open(output_path, 'wb') as f:
            pickle.dump(paper_embeddings, f)
        print(f"Saved all key embeddings to {output_path}")
        
        metadata = {
            "keys": self.embed_keys,
            "paper_count": len(paper_embeddings),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
        metadata_file = output_path.with_suffix(".meta.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        if not keep_backups:
            self.cleanup_backups()
        else:
            print(f"Kept {len(self.backup_files)} backup files as requested.")
        
        return paper_embeddings


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for individual paper keys using Qwen")
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Folder containing paper JSON files")
    parser.add_argument("--output_file", type=str, default="./embeddings/qwen_key_embeddings.pkl",
                        help="Output pickle file to save embeddings")
    parser.add_argument("--model_name", type=str, default="Alibaba-NLP/gte-Qwen2-7B-instruct",
                        help="Qwen embedding model name")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run model on ('cpu', 'cuda:0', etc.)")
    parser.add_argument("--keep_backups", action="store_true",
                        help="Keep backup files after processing completes")
    
    args = parser.parse_args()
    
    generator = QwenKeyEmbeddingGenerator(
        model_name=args.model_name,
        embeddings_path=os.path.dirname(args.output_file),
        device=args.device
    )
    
    generator.generate_key_embeddings(
        input_folder=args.input_folder,
        output_file=args.output_file,
        keep_backups=args.keep_backups
    )
    
    print("Embedding generation complete!")


if __name__ == "__main__":
    main()