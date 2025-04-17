import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from tqdm import tqdm
import torch
import pickle
from pathlib import Path

class PaperProcessor:
    def __init__(
        self,
        base_path: str,
        embedding_generator: str = "openai",
        summary_generator: str = "llama",
        clustering_method: str = "kmeans",
        clustering_direction: str = "bottom_up",
        openai_key: Optional[str] = None,
        cluster_prompt_file: Optional[str] = None,
        random_seed: int = 42,
        embeddings_subfolder: Optional[Path] = None,
        embedding_type: str = "all",
        embedding_key: Optional[str] = None,
        pre_generated_embeddings_file: Optional[str] = None
    ):
        """
        Initialize the paper processor with lazy loading of models.
        
        Args:
            base_path: Base directory for the project
            embedding_generator: Type of embedding generator to use
            summary_generator: Type of summary generator to use
            clustering_method: Type of clustering method to use
            clustering_direction: Direction of clustering ('bottom_up', 'top_down', or 'bidirectional')
            openai_key: OpenAI API key
            cluster_prompt_file: Path to cluster prompt file
            random_seed: Random seed for reproducibility
            embeddings_subfolder: Subfolder for embedding cache
            embedding_type: Type of embedding to use ("subkey", "key", or "all")
            embedding_key: Specific key to use (e.g., "problem" or "problem.overarching problem domain")
            pre_generated_embeddings_file: Path to pre-generated embeddings file
        """
        self.base_path = Path(base_path)
        self.results_path = self.base_path / "results"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.results_path.mkdir(exist_ok=True)

        self.embedding_generator_type = embedding_generator
        self.summary_generator_type = summary_generator
        self.clustering_method_type = clustering_method
        self.clustering_direction = clustering_direction
        self.openai_key = openai_key
        self.cluster_prompt_file = cluster_prompt_file
        self.random_seed = random_seed
        self.embeddings_path = embeddings_subfolder if embeddings_subfolder else (self.base_path / "embeddings")
        
        if self.clustering_direction not in ["bottom_up", "top_down", "bidirectional"]:
            raise ValueError("clustering_direction must be one of 'bottom_up', 'top_down', or 'bidirectional'")
        
        self.embedding_type = embedding_type.lower()
        self.embedding_key = embedding_key
        self.pre_generated_embeddings_file = pre_generated_embeddings_file
        self.pre_generated_embeddings = None
        
        if self.embedding_type not in ["subkey", "key", "all"]:
            raise ValueError("embedding_type must be one of 'subkey', 'key', or 'all'")
        
        if self.embedding_type == "subkey" and not self.embedding_key:
            raise ValueError("embedding_key must be specified when embedding_type is 'subkey'")
        if self.embedding_type == "key" and not self.embedding_key:
            raise ValueError("embedding_key must be specified when embedding_type is 'key'")
        
        if self.pre_generated_embeddings_file:
            print(f"Using pre-generated embeddings with {self.embedding_type} mode")
            if self.embedding_key:
                print(f"Using embedding key: {self.embedding_key}")
            
            self.load_pre_generated_embeddings()
        
        self._embedding_manager = None
        self._summarizer = None
        self._clustering = None

    @property
    def embedding_manager(self):
        """Lazy-load embedding manager when first accessed"""
        from .embedding import OpenAIEmbeddingGenerator, QwenEmbeddingGenerator
        
        if self._embedding_manager is None:
            print("Initializing embedding model...")
            if self.embedding_generator_type == "openai":
                self._embedding_manager = OpenAIEmbeddingGenerator(
                    openai_key=self.openai_key,
                    model_name="text-embedding-ada-002",
                    embeddings_path=self.base_path / "embeddings",
                    experiment_subfolder=self.embeddings_path
                )
            elif self.embedding_generator_type == "qwen":
                self._embedding_manager = QwenEmbeddingGenerator(
                    model_name="Alibaba-NLP/gte-Qwen2-7B-instruct",
                    embeddings_path=self.base_path / "embeddings",
                    experiment_subfolder=self.embeddings_path,
                )
            else:
                raise ValueError(f"Unsupported embedding_generator: {self.embedding_generator_type}")
        return self._embedding_manager
    
    @property
    def summarizer(self):
        """Lazy-load summarizer when first accessed"""
        from .summarizer import LlamaSummaryGenerator, GPTSummaryGenerator
        
        if self._summarizer is None:
            print("Initializing summarization model...")
            pf = Path(self.cluster_prompt_file) if self.cluster_prompt_file else None
            
            if self.summary_generator_type == "llama":
                self._summarizer = LlamaSummaryGenerator(
                    model_id="meta-llama/Llama-3.3-70B-Instruct",
                    prompt_file=pf
                )
            elif self.summary_generator_type == "gpt":
                self._summarizer = GPTSummaryGenerator(
                    model_id="gpt-4o-2024-08-06",
                    prompt_file=pf
                )
            else:
                raise ValueError(f"Unsupported summary_generator: {self.summary_generator_type}")
        return self._summarizer
    
    @property
    def clustering(self):
        """Lazy-load clustering when first accessed"""
        from .clustering import KMeansClustering, SpectralClust, AgglomerativeClust
        
        if self._clustering is None:
            print("Initializing clustering model...")
            
            if self.clustering_method_type == "kmeans":
                self._clustering = KMeansClustering(random_state=self.random_seed)
            elif self.clustering_method_type == "spectral":
                self._clustering = SpectralClust(random_state=self.random_seed)
            elif self.clustering_method_type == "agg":
                self._clustering = AgglomerativeClust(random_state=self.random_seed)
            else:
                raise ValueError(f"Unsupported clustering_method: {self.clustering_method_type}")
                
        return self._clustering

    def unload_models(self, keep_embeddings=False):
        """Unload models to free memory"""
        print("Unloading models to free memory...")
        
        if not keep_embeddings:
            if self._embedding_manager is not None:
                print("- Unloading embedding model")
                self._embedding_manager = None
        
        if self._summarizer is not None:
            print("- Unloading summarization model")
            try:
                self._summarizer.cleanup()
            except:
                pass
            self._summarizer = None
        
        if self._clustering is not None:
            print("- Unloading clustering model")
            self._clustering = None
        
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print("Memory cleanup completed")

    def load_pre_generated_embeddings(self):
        """Load pre-generated embeddings from file"""
        print(f"Loading pre-generated embeddings from {self.pre_generated_embeddings_file}")
        try:
            with open(self.pre_generated_embeddings_file, 'rb') as f:
                self.pre_generated_embeddings = pickle.load(f)
            print(f"Loaded embeddings for {len(self.pre_generated_embeddings)} papers")
        except Exception as e:
            raise RuntimeError(f"Failed to load pre-generated embeddings: {e}")
    
    def get_paper_embedding(self, paper_title: str, paper_data: Optional[dict] = None) -> np.ndarray:
        """
        Get embedding for a paper based on the selected embedding type and key.
        
        Args:
            paper_title: Title of the paper
            paper_data: Optional paper data (for logging only)
            
        Returns:
            numpy.ndarray: The paper embedding
        """
        if not self.pre_generated_embeddings:
            raise ValueError("Pre-generated embeddings not loaded")
        
        if paper_title not in self.pre_generated_embeddings:
            raise KeyError(f"Paper not found in pre-generated embeddings: {paper_title}")
        
        paper_embeddings = self.pre_generated_embeddings[paper_title]["key_embeddings"]
        
        if self.embedding_type == "subkey":
            if self.embedding_key not in paper_embeddings:
                raise KeyError(f"Subkey '{self.embedding_key}' not found for paper '{paper_title}'")
            return np.array(paper_embeddings[self.embedding_key]["embedding"])
        
        elif self.embedding_type == "key":
            matching_keys = [k for k in paper_embeddings.keys() if k.startswith(f"{self.embedding_key}.")]
            
            if not matching_keys:
                raise KeyError(f"No subkeys found for main key '{self.embedding_key}' in paper '{paper_title}'")
            
            combined_embedding = np.concatenate([
                paper_embeddings[k]["embedding"] for k in matching_keys
            ])
            
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
            
            return combined_embedding
        
        elif self.embedding_type == "all":
            all_embeddings = [
                paper_embeddings[k]["embedding"] for k in paper_embeddings.keys()
            ]
            
            combined_embedding = np.concatenate(all_embeddings)
            
            norm = np.linalg.norm(combined_embedding)
            if norm > 0:
                combined_embedding = combined_embedding / norm
            
            return combined_embedding
        
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def cleanup(self):
        """Clean up all resources"""
        self.unload_models()
        
        if hasattr(self, 'pre_generated_embeddings'):
            print("- Cleaning pre-generated embeddings...")
            self.pre_generated_embeddings = None
        
        print("PaperProcessor cleanup completed")

    def _load_papers(self, input_folder: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load papers from JSON files in the input folder
        
        Args:
            input_folder: Directory containing paper JSON files
            sample_size: If provided, use only this many randomly sampled papers
        
        Returns:
            DataFrame containing paper data
        """
        if not input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        files = list(input_folder.glob("*.json"))
        if not files:
            raise ValueError(f"No .json found in {input_folder}")
        
        if sample_size and sample_size < len(files):
            print(f"\nSampling {sample_size} files from total {len(files)} files...")
            random.seed(self.random_seed)
            files = random.sample(files, sample_size)
            print(f"Selected {len(files)} files for processing")

        all_papers = []
        for f_ in tqdm(files, desc="Reading papers"):
            try:
                data = json.loads(f_.read_text(encoding='utf-8'))
                papers_to_process = []
                
                if isinstance(data, dict):
                    if data.get('title') and (data.get('abstract') or data.get('problem')):
                        papers_to_process.append(data)
                elif isinstance(data, list):
                    papers_to_process.extend([
                        item for item in data 
                        if isinstance(item, dict) and item.get('title') and (item.get('abstract') or item.get('problem'))
                    ])

                for paper in papers_to_process:
                    paper_info = {
                        'file_name': f_.name,
                        'title': paper['title'].strip(),
                    }
                    
                    if 'abstract' in paper:
                        paper_info['abstract'] = paper['abstract'].strip()
                    
                    if 'problem' in paper:
                        paper_info['problem'] = paper['problem']
                    if 'solution' in paper:
                        paper_info['solution'] = paper['solution']
                    if 'results' in paper:
                        paper_info['results'] = paper['results']
                    
                    all_papers.append(paper_info)

            except Exception as e:
                print(f"[WARNING] error processing {f_}: {e}")

        if not all_papers:
            raise ValueError("No valid papers found in input folder")

        df = pd.DataFrame(all_papers)
        df.drop_duplicates(subset=['title'], inplace=True)
        
        return df
        
    def _allocate_subclusters(self, parent_clusters: Dict, total_subclusters: int) -> Dict[int, int]:
        """
        Allocate subclusters proportionally based on paper count in parent clusters.
        
        Args:
            parent_clusters: Dictionary of parent clusters {cluster_id: cluster_info}
            total_subclusters: Total number of subclusters to allocate
            
        Returns:
            Dict[int, int]: Allocation result as {parent_cluster_id: subcluster_count}
        """
        total_papers = 0
        for cluster in parent_clusters.values():
            total_papers += len(cluster['paper_ids'])
        
        if total_papers == 0:
            raise ValueError("No papers found in parent clusters")
        
        min_subclusters = len(parent_clusters)
        if total_subclusters < min_subclusters:
            print(f"Warning: total_subclusters ({total_subclusters}) < parent cluster count ({min_subclusters})")
            print(f"Setting each parent cluster to have exactly one subcluster")
            return {parent_id: 1 for parent_id in parent_clusters.keys()}
        
        remaining = total_subclusters - min_subclusters
        allocations = {}
        
        for parent_id, cluster in parent_clusters.items():
            paper_count = len(cluster['paper_ids'])
            proportion = paper_count / total_papers
            extra_allocation = max(0, int(round(proportion * remaining)))
            allocations[parent_id] = 1 + extra_allocation
        
        current_total = sum(allocations.values())
        adjustment = total_subclusters - current_total
        
        if adjustment != 0:
            sorted_clusters = sorted(
                parent_clusters.items(), 
                key=lambda x: len(x[1]['paper_ids']), 
                reverse=(adjustment > 0)
            )
            
            step = 1 if adjustment > 0 else -1
            i = 0
            while adjustment != 0:
                parent_id = sorted_clusters[i % len(sorted_clusters)][0]
                if step < 0 and allocations[parent_id] <= 1:
                    i += 1
                    continue
                allocations[parent_id] += step
                adjustment -= step
                i += 1
                
                if i > 1000:
                    print(f"Warning: Unable to exactly match target subcluster count. " 
                          f"Current: {sum(allocations.values())}, Target: {total_subclusters}")
                    break
        
        print(f"\nSubcluster allocation (target: {total_subclusters}):")
        for parent_id, count in allocations.items():
            papers = len(parent_clusters[parent_id]['paper_ids'])
            print(f"  Cluster {parent_id}: {count} subclusters for {papers} papers")
        print(f"Total subclusters: {sum(allocations.values())}")
        
        return allocations

    def process_hierarchical_clustering(
        self,
        cluster_sizes: List[int],
        run_seed: int,
        input_folder: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> dict:
        """
        Multi-level clustering with optimized memory usage.
        This is a unified interface that calls the appropriate clustering method.
        
        Args:
            cluster_sizes: List of cluster sizes for each level (from bottom to top)
            run_seed: Random seed for this run
            input_folder: Optional custom input folder path
            sample_size: If provided, use only this many randomly sampled papers
            
        Returns:
            dict: Hierarchical clustering result as a nested JSON structure
        """
        if self.clustering_direction == "bottom_up":
            print("Using bottom-up clustering approach")
            return self._process_bottom_up_clustering(
                cluster_sizes, run_seed, input_folder, sample_size
            )
        elif self.clustering_direction == "top_down":
            print("Using top-down clustering approach")
            return self._process_top_down_clustering(
                cluster_sizes, run_seed, input_folder, sample_size
            )
        else:
            print("Using bidirectional clustering approach")
            return self._process_bidirectional_clustering(
                cluster_sizes, run_seed, input_folder, sample_size
            )

    def _process_bottom_up_clustering(
        self,
        cluster_sizes: List[int],
        run_seed: int,
        input_folder: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> dict:
        """
        Bottom-up multi-level clustering with optimized memory usage.
        
        Args:
            cluster_sizes: List of cluster sizes for each level (from bottom to top)
            run_seed: Random seed for this run
            input_folder: Optional custom input folder path
            sample_size: If provided, use only this many randomly sampled papers
        """
        try:
            in_folder = Path(input_folder) if input_folder else (self.base_path / "dataset_10k")
            df = self._load_papers(in_folder, sample_size=sample_size)
            
            using_pregenerated = self.pre_generated_embeddings is not None
            
            if using_pregenerated:
                missing_papers = []
                for _, row in df.iterrows():
                    title = row['title']
                    if title not in self.pre_generated_embeddings:
                        missing_papers.append(title)
                
                if missing_papers:
                    error_msg = f"The following papers don't have pre-generated embeddings:\n"
                    for title in missing_papers[:5]:
                        error_msg += f"- {title}\n"
                    if len(missing_papers) > 5:
                        error_msg += f"... and {len(missing_papers) - 5} more.\n"
                    error_msg += "Please regenerate embeddings for all papers."
                    raise ValueError(error_msg)
                
                print(f"All {len(df)} papers found in pre-generated embeddings.")
                print(f"Using '{self.embedding_type}' embeddings" + 
                     (f" with key '{self.embedding_key}'" if self.embedding_key else ""))
            else:
                df['text_for_embedding'] = df.apply(
                    lambda x: f"Title: {x['title']}\n{x['abstract']}" if 'abstract' in x else None,
                    axis=1
                )

            hierarchy_info = {
                'papers': df,
                'levels': {}
            }

            for lvl, n_clusters in enumerate(cluster_sizes, start=1):
                print(f"\nProcessing level {lvl} with {n_clusters} clusters")
                self.random_seed = run_seed

                if lvl == 1:
                    if using_pregenerated:
                        paper_embeddings = []
                        for _, row in tqdm(df.iterrows(), desc="Getting paper embeddings", total=len(df)):
                            try:
                                paper_embedding = self.get_paper_embedding(row['title'], row)
                                paper_embeddings.append(paper_embedding)
                            except Exception as e:
                                raise RuntimeError(f"Error getting embedding for paper '{row['title']}': {e}")
                        
                        embeddings = np.array(paper_embeddings)
                        current_ids = list(range(len(df)))
                    else:
                        print("Loading embedding model for level 1...")
                        
                        current_texts = df['text_for_embedding'].tolist()
                        current_ids = list(range(len(df)))
                        
                        cache_file = self.base_path / "embeddings" / self.embeddings_path / f"level_{lvl}.pkl"
                        embeddings = self.embedding_manager.get_embeddings(
                            current_texts,
                            cache_file=cache_file,
                            is_paper=(lvl == 1)
                        )
                        
                        if len(cluster_sizes) > 1 and not using_pregenerated:
                            self.unload_models(keep_embeddings=False)
                else:
                    if using_pregenerated:
                        embeddings = np.array([
                            np.array(emb) for emb in current_cluster_embeddings
                        ])
                    else:
                        print(f"Loading embedding model for level {lvl}...")
                        
                        cache_file = self.base_path / "embeddings" / self.embeddings_path / f"level_{lvl}.pkl"
                        embeddings = self.embedding_manager.get_embeddings(
                            current_texts,
                            cache_file=cache_file,
                            is_paper=(lvl == 1)
                        )
                        
                        if not using_pregenerated:
                            self.unload_models(keep_embeddings=False)

                print(f"Running clustering for level {lvl}...")
                
                labels = self.clustering.perform_clustering(embeddings, n_clusters)
                
                self._clustering = None

                level_clusters = {}
                for i_, lbl_ in enumerate(labels):
                    if lbl_ not in level_clusters:
                        level_clusters[lbl_] = {
                            'paper_ids': [],
                            'name': None,
                            'summary': None,
                            'parent_cluster': None,
                            'child_clusters': []
                        }
                    level_clusters[lbl_]['paper_ids'].append(current_ids[i_])

                next_texts = []
                next_ids = []
                
                print("Loading summarization model...")
                
                for cid, info in tqdm(level_clusters.items(), desc="Generate summaries"):
                    if lvl == 1:
                        if using_pregenerated:
                            paper_indices = info['paper_ids']
                            paper_titles = [df.iloc[idx]['title'] for idx in paper_indices]
                            
                            c_texts = []
                            for title in paper_titles:
                                paper_data = self.pre_generated_embeddings[title]
                                text_parts = []
                                
                                if self.embedding_key == "problem":
                                    problem_keys = [k for k in paper_data["key_embeddings"] if k.startswith("problem.")]
                                    if problem_keys:
                                        problem_json = {}
                                        for key in problem_keys:
                                            key_name = key.replace("problem.", "")
                                            problem_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                        text_parts.append("Problem: " + json.dumps(problem_json, ensure_ascii=False))
                                
                                elif self.embedding_key == "solution":
                                    solution_keys = [k for k in paper_data["key_embeddings"] if k.startswith("solution.")]
                                    if solution_keys:
                                        solution_json = {}
                                        for key in solution_keys:
                                            key_name = key.replace("solution.", "")
                                            solution_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                        text_parts.append("Solution: " + json.dumps(solution_json, ensure_ascii=False))
                                
                                elif self.embedding_key == "results":
                                    results_keys = [k for k in paper_data["key_embeddings"] if k.startswith("results.")]
                                    if results_keys:
                                        results_json = {}
                                        for key in results_keys:
                                            key_name = key.replace("results.", "")
                                            results_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                        text_parts.append("Results: " + json.dumps(results_json, ensure_ascii=False))
                                
                                c_texts.append("Title: " + title + "\n" + "\n".join(text_parts))
                        else:
                            c_texts = [df.iloc[idx]['text_for_embedding'] for idx in info['paper_ids']]
                    else:
                        c_texts = [current_texts[j] for j, lab in enumerate(labels) if lab == cid]
                    
                    cname, csummary = self.summarizer.generate_cluster_summary(c_texts)
                    info['name'] = cname
                    info['summary'] = csummary

                    next_text = f"ClusterName: {cname}\nClusterSummary: {csummary}"
                    next_texts.append(next_text)
                    next_ids.append(cid)
                
                if hasattr(self, '_summarizer') and self._summarizer is not None:
                    print(f"Unloading summarizer after level {lvl}...")
                    try:
                        self._summarizer.cleanup()
                    except:
                        pass
                    self._summarizer = None
                    
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                if using_pregenerated and next_texts:
                    print(f"Loading embedding model for summary embeddings at level {lvl}...")
                    
                    cache_file = self.base_path / "embeddings" / self.embeddings_path / f"level_{lvl+1}.pkl"
                    
                    print(f"Generating embeddings for {len(next_texts)} cluster summaries at level {lvl+1}")
                    summary_embeddings = self.embedding_manager.get_embeddings(
                        next_texts,
                        cache_file=cache_file,
                        is_paper=False
                    )
                    
                    current_cluster_embeddings = summary_embeddings
                    
                    self.unload_models(keep_embeddings=False)

                hierarchy_info['levels'][lvl] = level_clusters

                current_texts = next_texts
                current_ids = next_ids

            max_level = len(cluster_sizes)
            for lvl in range(1, max_level):
                curr_level_clusters = hierarchy_info['levels'][lvl]
                next_level_clusters = hierarchy_info['levels'][lvl + 1]

                for cid, cinfo in next_level_clusters.items():
                    cinfo['child_clusters'] = cinfo['paper_ids']
                    del cinfo['paper_ids']
                    for child_id in cinfo['child_clusters']:
                        curr_level_clusters[child_id]['parent_cluster'] = cid

            final_json = self._build_nested_json(hierarchy_info)
            return final_json
        finally:
            self.cleanup()

    def _process_top_down_clustering(
        self,
        cluster_sizes: List[int],
        run_seed: int,
        input_folder: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> dict:
        """
        Top-down multi-level clustering with optimized memory usage.
        Uses proportional allocation of subclusters.
        
        Args:
            cluster_sizes: List of cluster sizes for each level (from bottom to top)
            run_seed: Random seed for this run
            input_folder: Optional custom input folder path
            sample_size: If provided, use only this many randomly sampled papers
        """
        try:
            in_folder = Path(input_folder) if input_folder else (self.base_path / "dataset_10k")
            df = self._load_papers(in_folder, sample_size=sample_size)
            
            using_pregenerated = self.pre_generated_embeddings is not None
            
            if using_pregenerated:
                missing_papers = []
                for _, row in df.iterrows():
                    title = row['title']
                    if title not in self.pre_generated_embeddings:
                        missing_papers.append(title)
                
                if missing_papers:
                    error_msg = f"The following papers don't have pre-generated embeddings:\n"
                    for title in missing_papers[:5]:
                        error_msg += f"- {title}\n"
                    if len(missing_papers) > 5:
                        error_msg += f"... and {len(missing_papers) - 5} more.\n"
                    error_msg += "Please regenerate embeddings for all papers."
                    raise ValueError(error_msg)
                
                print(f"All {len(df)} papers found in pre-generated embeddings.")
                print(f"Using '{self.embedding_type}' embeddings" + 
                     (f" with key '{self.embedding_key}'" if self.embedding_key else ""))
            else:
                df['text_for_embedding'] = df.apply(
                    lambda x: f"Title: {x['title']}\n{x['abstract']}" if 'abstract' in x else None,
                    axis=1
                )

            hierarchy_info = {
                'papers': df,
                'levels': {}
            }
            
            reversed_cluster_sizes = list(reversed(cluster_sizes))
            max_level = len(reversed_cluster_sizes)
            
            if using_pregenerated:
                paper_embeddings = []
                for _, row in tqdm(df.iterrows(), desc="Getting paper embeddings", total=len(df)):
                    try:
                        paper_embedding = self.get_paper_embedding(row['title'], row)
                        paper_embeddings.append(paper_embedding)
                    except Exception as e:
                        raise RuntimeError(f"Error getting embedding for paper '{row['title']}': {e}")
                
                all_paper_embeddings = np.array(paper_embeddings)
            else:
                texts_for_embedding = df['text_for_embedding'].tolist()
                
                cache_file = self.base_path / "embeddings" / self.embeddings_path / "papers.pkl"
                all_paper_embeddings = self.embedding_manager.get_embeddings(
                    texts_for_embedding,
                    cache_file=cache_file,
                    is_paper=True
                )
                
                if not using_pregenerated:
                    self.unload_models(keep_embeddings=False)
            
            all_paper_ids = list(range(len(df)))
            
            paper_to_cluster_map = {}
            
            for lvl, n_clusters in enumerate(reversed_cluster_sizes, start=1):
                print(f"\nProcessing level {lvl} with {n_clusters} clusters (top-down)")
                self.random_seed = run_seed
                
                level_clusters = {}
                
                level_paper_to_cluster = {}
                
                if lvl == 1:
                    labels = self.clustering.perform_clustering(all_paper_embeddings, n_clusters)
                    
                    for i_, lbl_ in enumerate(labels):
                        if lbl_ not in level_clusters:
                            level_clusters[lbl_] = {
                                'paper_ids': [],
                                'name': None,
                                'summary': None,
                                'parent_cluster': None,
                                'child_clusters': []
                            }
                        
                        level_clusters[lbl_]['paper_ids'].append(all_paper_ids[i_])
                        level_paper_to_cluster[all_paper_ids[i_]] = lbl_
                else:
                    prev_level = lvl - 1
                    prev_clusters = hierarchy_info['levels'][prev_level]
                    
                    subclusters_allocation = self._allocate_subclusters(prev_clusters, n_clusters)
                    
                    next_cluster_id = 0
                    
                    for parent_id, subcluster_count in subclusters_allocation.items():
                        parent_cluster = prev_clusters[parent_id]
                        papers_in_cluster = parent_cluster['paper_ids']
                        
                        if len(papers_in_cluster) <= 1 or subcluster_count <= 1:
                            cluster_id = next_cluster_id
                            level_clusters[cluster_id] = {
                                'paper_ids': papers_in_cluster.copy(),
                                'name': None,
                                'summary': None,
                                'parent_cluster': parent_id,
                                'child_clusters': []
                            }
                            
                            for pid in papers_in_cluster:
                                level_paper_to_cluster[pid] = cluster_id
                                
                            parent_cluster['child_clusters'].append(cluster_id)
                            
                            next_cluster_id += 1
                            continue
                        
                        cluster_embeddings = np.array([all_paper_embeddings[pid] for pid in papers_in_cluster])
                        
                        sub_labels = self.clustering.perform_clustering(cluster_embeddings, subcluster_count)
                        
                        sub_clusters_map = {}
                        for i_, sub_lbl in enumerate(sub_labels):
                            if sub_lbl not in sub_clusters_map:
                                cluster_id = next_cluster_id
                                sub_clusters_map[sub_lbl] = cluster_id
                                level_clusters[cluster_id] = {
                                    'paper_ids': [],
                                    'name': None,
                                    'summary': None,
                                    'parent_cluster': parent_id,
                                    'child_clusters': []
                                }
                                
                                parent_cluster['child_clusters'].append(cluster_id)
                                
                                next_cluster_id += 1
                            
                            paper_id = papers_in_cluster[i_]
                            cluster_id = sub_clusters_map[sub_lbl]
                            level_clusters[cluster_id]['paper_ids'].append(paper_id)
                            level_paper_to_cluster[paper_id] = cluster_id
                
                self._clustering = None
                
                print("Loading summarization model...")
                for cid, info in tqdm(level_clusters.items(), desc="Generate summaries"):
                    paper_indices = info['paper_ids']
                    
                    if using_pregenerated:
                        paper_titles = [df.iloc[pid]['title'] for pid in paper_indices]
                        
                        c_texts = []
                        for title in paper_titles:
                            paper_data = self.pre_generated_embeddings[title]
                            text_parts = []
                            
                            if self.embedding_key == "problem":
                                problem_keys = [k for k in paper_data["key_embeddings"] if k.startswith("problem.")]
                                if problem_keys:
                                    problem_json = {}
                                    for key in problem_keys:
                                        key_name = key.replace("problem.", "")
                                        problem_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                    text_parts.append("Problem: " + json.dumps(problem_json, ensure_ascii=False))
                            
                            elif self.embedding_key == "solution":
                                solution_keys = [k for k in paper_data["key_embeddings"] if k.startswith("solution.")]
                                if solution_keys:
                                    solution_json = {}
                                    for key in solution_keys:
                                        key_name = key.replace("solution.", "")
                                        solution_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                    text_parts.append("Solution: " + json.dumps(solution_json, ensure_ascii=False))
                            
                            elif self.embedding_key == "results":
                                results_keys = [k for k in paper_data["key_embeddings"] if k.startswith("results.")]
                                if results_keys:
                                    results_json = {}
                                    for key in results_keys:
                                        key_name = key.replace("results.", "")
                                        results_json[key_name] = paper_data["key_embeddings"][key]["text"]
                                    text_parts.append("Results: " + json.dumps(results_json, ensure_ascii=False))
                            
                            c_texts.append("Title: " + title + "\n" + "\n".join(text_parts))
                    else:
                        c_texts = [df.iloc[pid]['text_for_embedding'] for pid in paper_indices]
                    
                    cname, csummary = self.summarizer.generate_cluster_summary(c_texts)
                    info['name'] = cname
                    info['summary'] = csummary
                
                if hasattr(self, '_summarizer') and self._summarizer is not None:
                    print(f"Unloading summarizer after level {lvl}...")
                    try:
                        self._summarizer.cleanup()
                    except:
                        pass
                    self._summarizer = None
                    
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                
                hierarchy_info['levels'][lvl] = level_clusters
                paper_to_cluster_map[lvl] = level_paper_to_cluster

            final_json = self._build_nested_json_top_down(hierarchy_info, max_level)
            return final_json
        finally:
            self.cleanup()

    def _process_bidirectional_clustering(
        self,
        cluster_sizes: List[int],
        run_seed: int,
        input_folder: Optional[str] = None,
        sample_size: Optional[int] = None,
    ) -> dict:
        """
        Bidirectional clustering: top-down for top level, bottom-up within each top cluster.
        Optimized for GPU memory efficiency with correct model loading/unloading sequence.
        
        Args:
            cluster_sizes: List of cluster sizes for each level (from bottom to top)
            run_seed: Random seed for this run
            input_folder: Optional custom input folder path
            sample_size: If provided, use only this many randomly sampled papers
        """
        try:
            print("\n=== STARTING BIDIRECTIONAL CLUSTERING (GPU MEMORY OPTIMIZED) ===")
            
            in_folder = Path(input_folder) if input_folder else (self.base_path / "dataset_10k")
            df = self._load_papers(in_folder, sample_size=sample_size)
            
            using_pregenerated = self.pre_generated_embeddings is not None
            
            hierarchy_info = {
                'papers': df,
                'levels': {}
            }
            
            bottom_level_size = cluster_sizes[0]
            middle_level_size = cluster_sizes[1]
            top_level_size = cluster_sizes[2]
            
            if not using_pregenerated:
                df['text_for_embedding'] = df.apply(
                    lambda x: f"Title: {x['title']}\n{x['abstract']}" if 'abstract' in x else None,
                    axis=1
                )
            
            print("\n=== PHASE 1: PREPARING PAPER EMBEDDINGS ===")
            
            all_paper_ids = list(range(len(df)))
            if using_pregenerated:
                missing_papers = []
                for _, row in df.iterrows():
                    title = row['title']
                    if title not in self.pre_generated_embeddings:
                        missing_papers.append(title)
                
                if missing_papers:
                    error_msg = f"Missing pre-generated embeddings for {len(missing_papers)} papers"
                    raise ValueError(error_msg)
                
                print(f"Using pre-generated embeddings for {len(df)} papers")
                
                paper_embeddings = []
                for _, row in tqdm(df.iterrows(), desc="Getting embeddings"):
                    paper_embedding = self.get_paper_embedding(row['title'], row)
                    paper_embeddings.append(paper_embedding)
                
                all_paper_embeddings = np.array(paper_embeddings)
            else:
                texts_for_embedding = df['text_for_embedding'].tolist()
                
                cache_file = self.base_path / "embeddings" / self.embeddings_path / "papers.pkl"
                all_paper_embeddings = self.embedding_manager.get_embeddings(
                    texts_for_embedding,
                    cache_file=cache_file,
                    is_paper=True
                )
                
                print("Unloading embedding model...")
                self.unload_models(keep_embeddings=False)
            
            print(f"\n=== PHASE 2: TOP-DOWN CLUSTERING FOR TOP LEVEL (Size: {top_level_size}) ===")
            
            self.random_seed = run_seed
            self._clustering = None
            top_level_labels = self.clustering.perform_clustering(all_paper_embeddings, top_level_size)
            
            self._clustering = None
            import gc
            gc.collect()
            
            top_level_clusters = {}
            
            for i, label in enumerate(top_level_labels):
                paper_id = all_paper_ids[i]
                
                if label not in top_level_clusters:
                    top_level_clusters[label] = {
                        'paper_ids': [],
                        'name': None,
                        'summary': None,
                        'parent_cluster': None,
                        'child_clusters': []
                    }
                
                top_level_clusters[label]['paper_ids'].append(paper_id)
            
            total_papers = len(df)
            cluster_allocations = {}
            
            print("\n=== CLUSTER ALLOCATION BY PAPER PROPORTION ===")
            print(f"Bottom level: {bottom_level_size} clusters total")
            print(f"Middle level: {middle_level_size} clusters total")
            
            for top_id, top_info in top_level_clusters.items():
                papers_in_cluster = len(top_info['paper_ids'])
                proportion = papers_in_cluster / total_papers
                
                bottom_count = max(1, round(bottom_level_size * proportion))
                middle_count = max(1, round(middle_level_size * proportion))
                
                cluster_allocations[top_id] = {
                    'bottom': bottom_count,
                    'middle': middle_count,
                    'paper_count': papers_in_cluster,
                    'proportion': proportion
                }
                
                print(f"Top cluster {top_id}: {papers_in_cluster} papers ({proportion:.1%}) → {bottom_count} bottom, {middle_count} middle clusters")
            
            self._adjust_cluster_allocation(cluster_allocations, bottom_level_size, middle_level_size)
            
            print("\n=== PHASE 3: GENERATING SUMMARIES FOR TOP LEVEL CLUSTERS ===")
            
            print("Loading summarization model...")
            
            for top_id, top_info in tqdm(top_level_clusters.items(), desc="Generating top level summaries"):
                paper_ids = top_info['paper_ids']
                
                if using_pregenerated:
                    paper_titles = [df.iloc[pid]['title'] for pid in paper_ids]
                    c_texts = self._get_texts_from_pregenerated(paper_titles)
                else:
                    c_texts = [df.iloc[pid]['text_for_embedding'] for pid in paper_ids]
                
                cname, csummary = self.summarizer.generate_cluster_summary(c_texts)
                top_info['name'] = cname
                top_info['summary'] = csummary
            
            print("Unloading summarization model...")
            if hasattr(self, '_summarizer') and self._summarizer is not None:
                try:
                    self._summarizer.cleanup()
                except:
                    pass
                self._summarizer = None
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            max_level = len(cluster_sizes)
            hierarchy_info['levels'][max_level] = top_level_clusters
            
            if 2 not in hierarchy_info['levels']:
                hierarchy_info['levels'][2] = {}
            
            if 1 not in hierarchy_info['levels']:
                hierarchy_info['levels'][1] = {}
            
            print(f"\n=== PHASE 4: BOTTOM-UP CLUSTERING WITHIN TOP CLUSTERS ===")
            
            print("\n=== PHASE 4.1: CREATING BOTTOM LEVEL CLUSTERS ===")
            
            all_bottom_clusters = {}
            
            for top_id, top_info in tqdm(top_level_clusters.items(), desc="Creating bottom clusters"):
                papers_in_cluster = top_info['paper_ids']
                allocation = cluster_allocations[top_id]
                
                cluster_embeddings = all_paper_embeddings[papers_in_cluster]
                
                self.random_seed = run_seed
                bottom_cluster_count = allocation['bottom']
                actual_bottom_count = min(bottom_cluster_count, len(papers_in_cluster))
                
                self._clustering = None
                
                bottom_labels = self.clustering.perform_clustering(cluster_embeddings, actual_bottom_count)
                
                self._clustering = None
                
                bottom_level_clusters = {}
                
                for i, label in enumerate(bottom_labels):
                    bottom_id = f"{top_id}_b{label}"
                    paper_id = papers_in_cluster[i]
                    
                    if bottom_id not in bottom_level_clusters:
                        bottom_level_clusters[bottom_id] = {
                            'paper_ids': [],
                            'name': None,
                            'summary': None,
                            'parent_cluster': None,
                            'child_clusters': []
                        }
                    
                    bottom_level_clusters[bottom_id]['paper_ids'].append(paper_id)
                
                all_bottom_clusters[top_id] = bottom_level_clusters
            
            del all_paper_embeddings
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("\n=== PHASE 4.2: GENERATING BOTTOM LEVEL SUMMARIES ===")
            
            print("Loading summarization model...")
            
            for top_id, bottom_clusters in tqdm(all_bottom_clusters.items(), desc="Summarizing bottom clusters by top cluster"):
                for bottom_id, info in tqdm(bottom_clusters.items(), desc=f"Bottom clusters in top {top_id}", leave=False):
                    cluster_paper_ids = info['paper_ids']
                    
                    if using_pregenerated:
                        paper_titles = [df.iloc[pid]['title'] for pid in cluster_paper_ids]
                        c_texts = self._get_texts_from_pregenerated(paper_titles)
                    else:
                        c_texts = [df.iloc[pid]['text_for_embedding'] for pid in cluster_paper_ids]
                    
                    cname, csummary = self.summarizer.generate_cluster_summary(c_texts)
                    info['name'] = cname
                    info['summary'] = csummary
            
            print("Unloading summarization model...")
            if hasattr(self, '_summarizer') and self._summarizer is not None:
                try:
                    self._summarizer.cleanup()
                except:
                    pass
                self._summarizer = None
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("\n=== PHASE 4.3: GENERATING BOTTOM SUMMARY EMBEDDINGS AND CREATING MIDDLE CLUSTERS ===")
            
            print("Loading embedding model for summary embeddings...")
            
            all_middle_clusters = {}
            
            for top_id, bottom_clusters in tqdm(all_bottom_clusters.items(), desc="Processing top clusters"):
                summary_texts = []
                bottom_ids = []
                
                for bottom_id, info in bottom_clusters.items():
                    summary_text = f"ClusterName: {info['name']}\nClusterSummary: {info['summary']}"
                    summary_texts.append(summary_text)
                    bottom_ids.append(bottom_id)
                
                if len(summary_texts) <= 1:
                    print(f"Warning: Top cluster {top_id} has only {len(summary_texts)} bottom clusters, creating single middle cluster")
                    middle_id = f"{top_id}_m0"
                    middle_clusters = {
                        middle_id: {
                            'name': None,
                            'summary': None,
                            'parent_cluster': top_id,
                            'child_clusters': list(bottom_clusters.keys())
                        }
                    }
                    
                    for bottom_id in bottom_clusters:
                        bottom_clusters[bottom_id]['parent_cluster'] = middle_id
                    
                    all_middle_clusters[top_id] = middle_clusters
                    continue
                
                cache_file = self.base_path / "embeddings" / self.embeddings_path / f"bottom_summaries_{top_id}.pkl"
                summary_embeddings = self.embedding_manager.get_embeddings(
                    summary_texts,
                    cache_file=cache_file,
                    is_paper=False
                )
                
                middle_count = cluster_allocations[top_id]['middle']
                actual_middle_count = min(middle_count, len(bottom_ids))
                
                self.random_seed = run_seed
                self._clustering = None
                middle_labels = self.clustering.perform_clustering(summary_embeddings, actual_middle_count)
                self._clustering = None
                
                middle_clusters = {}
                
                for i, label in enumerate(middle_labels):
                    middle_id = f"{top_id}_m{label}"
                    bottom_id = bottom_ids[i]
                    
                    if middle_id not in middle_clusters:
                        middle_clusters[middle_id] = {
                            'name': None,
                            'summary': None,
                            'parent_cluster': top_id,
                            'child_clusters': []
                        }
                    
                    if bottom_id not in middle_clusters[middle_id]['child_clusters']:
                        middle_clusters[middle_id]['child_clusters'].append(bottom_id)
                    
                    bottom_clusters[bottom_id]['parent_cluster'] = middle_id
                
                all_middle_clusters[top_id] = middle_clusters
            
            print("Unloading embedding model...")
            self.unload_models(keep_embeddings=False)
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("\n=== PHASE 4.4: GENERATING MIDDLE LEVEL SUMMARIES ===")
            
            print("Loading summarization model...")
            
            for top_id, middle_clusters in tqdm(all_middle_clusters.items(), desc="Summarizing middle clusters by top cluster"):
                bottom_clusters = all_bottom_clusters[top_id]
                
                for middle_id, info in tqdm(middle_clusters.items(), desc=f"Middle clusters in top {top_id}", leave=False):
                    child_texts = []
                    for child_id in info['child_clusters']:
                        child_info = bottom_clusters[child_id]
                        child_name = child_info['name']
                        child_summary = child_info['summary']
                        child_texts.append(f"ClusterName: {child_name}\nClusterSummary: {child_summary}")
                    
                    if child_texts:
                        cname, csummary = self.summarizer.generate_cluster_summary(child_texts)
                        info['name'] = cname
                        info['summary'] = csummary
                    else:
                        info['name'] = f"Middle Cluster {middle_id}"
                        info['summary'] = "No summary available due to empty child clusters"
            
            print("Unloading summarization model...")
            if hasattr(self, '_summarizer') and self._summarizer is not None:
                try:
                    self._summarizer.cleanup()
                except:
                    pass
                self._summarizer = None
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            print("\n=== PHASE 5: MERGING ALL CLUSTERS INTO HIERARCHY ===")
            
            for top_id, bottom_clusters in all_bottom_clusters.items():
                for bottom_id, info in bottom_clusters.items():
                    hierarchy_info['levels'][1][bottom_id] = info
            
            for top_id, middle_clusters in all_middle_clusters.items():
                for middle_id, info in middle_clusters.items():
                    top_level_clusters[top_id]['child_clusters'].append(middle_id)
                    hierarchy_info['levels'][2][middle_id] = info
            
            print("\n=== PHASE 6: REMAPPING CLUSTER IDs FOR CLEANER PATHS ===")
            hierarchy_info = self._remap_cluster_ids(hierarchy_info)
            
            print("\n=== PHASE 7: BUILDING FINAL HIERARCHY JSON ===")
            final_json = self._build_nested_json(hierarchy_info)
            
            return final_json
        finally:
            self.cleanup()

    def _process_bottom_up_within_top_cluster(
        self,
        paper_ids: List[int],
        bottom_cluster_count: int,
        middle_cluster_count: int,
        df: pd.DataFrame,
        all_paper_embeddings: np.ndarray,
        using_pregenerated: bool,
        run_seed: int,
        top_cluster_id: int
    ) -> Tuple[Dict, Dict]:
        """
        Perform bottom-up clustering within a top level cluster.
        
        Args:
            paper_ids: List of paper IDs in this top cluster
            bottom_cluster_count: Number of bottom level clusters to create
            middle_cluster_count: Number of middle level clusters to create
            df: DataFrame containing paper data
            all_paper_embeddings: Array of all paper embeddings
            using_pregenerated: Whether using pre-generated embeddings
            run_seed: Random seed
            top_cluster_id: ID of the top level cluster
            
        Returns:
            Tuple[Dict, Dict]: (middle_level_clusters, bottom_level_clusters)
        """
        print(f"Bottom-up clustering with {bottom_cluster_count} bottom clusters, {middle_cluster_count} middle clusters")
        
        cluster_embeddings = all_paper_embeddings[paper_ids]
        
        self.random_seed = run_seed
        self._clustering = None
        
        actual_bottom_count = min(bottom_cluster_count, len(paper_ids))
        if actual_bottom_count < bottom_cluster_count:
            print(f"Warning: Reduced bottom clusters from {bottom_cluster_count} to {actual_bottom_count} due to paper count")
        
        bottom_labels = self.clustering.perform_clustering(cluster_embeddings, actual_bottom_count)
        self._clustering = None
        
        bottom_level_clusters = {}
        paper_to_bottom = {}
        
        for i, label in enumerate(bottom_labels):
            bottom_id = f"{top_cluster_id}_b{label}"
            paper_id = paper_ids[i]
            paper_to_bottom[paper_id] = bottom_id
            
            if bottom_id not in bottom_level_clusters:
                bottom_level_clusters[bottom_id] = {
                    'paper_ids': [],
                    'name': None,
                    'summary': None,
                    'parent_cluster': None,
                    'child_clusters': []
                }
            
            bottom_level_clusters[bottom_id]['paper_ids'].append(paper_id)
        
        print(f"Generating summaries for {len(bottom_level_clusters)} bottom clusters")
        bottom_summaries = {}
        
        for bottom_id, info in tqdm(bottom_level_clusters.items()):
            cluster_paper_ids = info['paper_ids']
            
            if using_pregenerated:
                paper_titles = [df.iloc[pid]['title'] for pid in cluster_paper_ids]
                c_texts = self._get_texts_from_pregenerated(paper_titles)
            else:
                c_texts = [df.iloc[pid]['text_for_embedding'] for pid in cluster_paper_ids]
            
            cname, csummary = self.summarizer.generate_cluster_summary(c_texts)
            info['name'] = cname
            info['summary'] = csummary
            
            bottom_summaries[bottom_id] = (cname, csummary)
        
        summary_texts = []
        bottom_ids = []
        
        for bottom_id, (name, summary) in bottom_summaries.items():
            summary_text = f"ClusterName: {name}\nClusterSummary: {summary}"
            summary_texts.append(summary_text)
            bottom_ids.append(bottom_id)
        
        summary_embeddings = self.embedding_manager.get_embeddings(
            summary_texts,
            cache_file=None,
            is_paper=False
        )
        
        actual_middle_count = min(middle_cluster_count, len(bottom_ids))
        if actual_middle_count < middle_cluster_count:
            print(f"Warning: Reduced middle clusters from {middle_cluster_count} to {actual_middle_count} due to bottom cluster count")
        
        self._clustering = None
        middle_labels = self.clustering.perform_clustering(summary_embeddings, actual_middle_count)
        self._clustering = None
        
        middle_level_clusters = {}
        
        for i, label in enumerate(middle_labels):
            middle_id = f"{top_cluster_id}_m{label}"
            bottom_id = bottom_ids[i]
            
            if middle_id not in middle_level_clusters:
                middle_level_clusters[middle_id] = {
                    'paper_ids': [],
                    'name': None,
                    'summary': None,
                    'parent_cluster': top_cluster_id,
                    'child_clusters': []
                }
            
            if bottom_id not in middle_level_clusters[middle_id]['child_clusters']:
                middle_level_clusters[middle_id]['child_clusters'].append(bottom_id)
            
            bottom_level_clusters[bottom_id]['parent_cluster'] = middle_id
        
        print(f"Generating summaries for {len(middle_level_clusters)} middle clusters")
        for middle_id, info in tqdm(middle_level_clusters.items()):
            child_texts = []
            for child_id in info['child_clusters']:
                child_name = bottom_level_clusters[child_id]['name']
                child_summary = bottom_level_clusters[child_id]['summary']
                child_texts.append(f"ClusterName: {child_name}\nClusterSummary: {child_summary}")
            
            cname, csummary = self.summarizer.generate_cluster_summary(child_texts)
            info['name'] = cname
            info['summary'] = csummary
        
        return middle_level_clusters, bottom_level_clusters

    def _adjust_cluster_allocation(
        self, 
        allocations: Dict, 
        target_bottom: int, 
        target_middle: int
    ):
        """
        Adjust cluster allocations to ensure totals match target values.
        
        Args:
            allocations: Dictionary of allocations by top cluster
            target_bottom: Target number of bottom level clusters
            target_middle: Target number of middle level clusters
        """
        current_bottom = sum(a['bottom'] for a in allocations.values())
        current_middle = sum(a['middle'] for a in allocations.values())
        
        self._adjust_level_allocation(allocations, 'bottom', current_bottom, target_bottom)
        
        self._adjust_level_allocation(allocations, 'middle', current_middle, target_middle)
        
        print("\nFinal cluster allocation:")
        bottom_sum = 0
        middle_sum = 0
        for top_id, alloc in allocations.items():
            print(f"Top cluster {top_id}: {alloc['bottom']} bottom, {alloc['middle']} middle clusters")
            bottom_sum += alloc['bottom']
            middle_sum += alloc['middle']
        
        print(f"Total: {bottom_sum} bottom clusters, {middle_sum} middle clusters")

    def _adjust_level_allocation(
        self,
        allocations: Dict,
        level_key: str,
        current_count: int,
        target_count: int
    ):
        """
        Adjust allocation for a specific level.
        
        Args:
            allocations: Dictionary of allocations
            level_key: Key for the level being adjusted ('bottom' or 'middle')
            current_count: Current total count
            target_count: Target total count
        """
        if current_count == target_count:
            return
        
        adjustment = target_count - current_count
        step = 1 if adjustment > 0 else -1
        
        sorted_clusters = sorted(
            allocations.items(),
            key=lambda x: x[1]['proportion'],
            reverse=(step > 0)
        )
        
        i = 0
        while adjustment != 0:
            cluster_id, cluster_info = sorted_clusters[i % len(sorted_clusters)]
            
            if step < 0 and allocations[cluster_id][level_key] <= 1:
                i += 1
                continue
            
            allocations[cluster_id][level_key] += step
            adjustment -= step
            i += 1

    def _get_texts_from_pregenerated(self, paper_titles: List[str]) -> List[str]:
        """
        Extract text content from pre-generated embeddings.
        
        Args:
            paper_titles: List of paper titles
            
        Returns:
            List[str]: List of text content for summarization
        """
        c_texts = []
        for title in paper_titles:
            paper_data = self.pre_generated_embeddings[title]
            text_parts = []
            
            if self.embedding_key == "problem":
                problem_keys = [k for k in paper_data["key_embeddings"] if k.startswith("problem.")]
                if problem_keys:
                    problem_json = {}
                    for key in problem_keys:
                        key_name = key.replace("problem.", "")
                        problem_json[key_name] = paper_data["key_embeddings"][key]["text"]
                    text_parts.append("Problem: " + json.dumps(problem_json, ensure_ascii=False))
            
            elif self.embedding_key == "solution":
                solution_keys = [k for k in paper_data["key_embeddings"] if k.startswith("solution.")]
                if solution_keys:
                    solution_json = {}
                    for key in solution_keys:
                        key_name = key.replace("solution.", "")
                        solution_json[key_name] = paper_data["key_embeddings"][key]["text"]
                    text_parts.append("Solution: " + json.dumps(solution_json, ensure_ascii=False))
            
            elif self.embedding_key == "results":
                results_keys = [k for k in paper_data["key_embeddings"] if k.startswith("results.")]
                if results_keys:
                    results_json = {}
                    for key in results_keys:
                        key_name = key.replace("results.", "")
                        results_json[key_name] = paper_data["key_embeddings"][key]["text"]
                    text_parts.append("Results: " + json.dumps(results_json, ensure_ascii=False))
            
            c_texts.append("Title: " + title + "\n" + "\n".join(text_parts))
        
        return c_texts

    def _remap_cluster_ids(self, hierarchy_info: Dict) -> Dict:
        """
        Remap cluster IDs to create cleaner paths.
        Converts paths like [1, '1_0', '1_7'] to [1, 2, 3].
        
        Args:
            hierarchy_info: Existing hierarchy information
            
        Returns:
            Dict: Updated hierarchy with remapped IDs
        """
        levels = sorted(hierarchy_info['levels'].keys())
        
        id_mappings = {}
        new_levels = {}
        
        for level in levels:
            level_clusters = hierarchy_info['levels'][level]
            id_mappings[level] = {}
            new_levels[level] = {}
            
            for new_id, old_id in enumerate(sorted(level_clusters.keys())):
                id_mappings[level][old_id] = new_id
                new_levels[level][new_id] = level_clusters[old_id].copy()
        
        for level in levels:
            for new_id, cluster_info in new_levels[level].items():
                if cluster_info['parent_cluster'] is not None:
                    parent_level = level + 1
                    old_parent_id = cluster_info['parent_cluster']
                    if parent_level in id_mappings and old_parent_id in id_mappings[parent_level]:
                        cluster_info['parent_cluster'] = id_mappings[parent_level][old_parent_id]
                    else:
                        cluster_info['parent_cluster'] = None
                        print(f"Warning: Parent cluster {old_parent_id} at level {parent_level} not found in mappings")
                
                new_child_clusters = []
                for old_child_id in cluster_info['child_clusters']:
                    child_level = level - 1
                    if child_level in id_mappings and old_child_id in id_mappings[child_level]:
                        new_child_clusters.append(id_mappings[child_level][old_child_id])
                    else:
                        print(f"Warning: Child cluster {old_child_id} at level {child_level} not found in mappings")
                cluster_info['child_clusters'] = new_child_clusters
        
        remapped_hierarchy = {
            'papers': hierarchy_info['papers'],
            'levels': new_levels
        }
        
        print(f"Remapped cluster IDs across {len(levels)} levels")
        return remapped_hierarchy


    def _build_nested_json(self, hierarchy_info: Dict) -> Dict:
        """
        Build a nested JSON from the top level down to the papers at level=1.
        For bottom-up approach.
        """
        max_level = max(hierarchy_info['levels'].keys())
        df = hierarchy_info['papers']

        def build_node(level: int, cid: int) -> Dict:
            cluster_info = hierarchy_info['levels'][level][cid]
            node = {
                "cluster_id": cid,
                "title": cluster_info['name'],
                "abstract": cluster_info['summary']
            }
            if level == 1:
                papers = []
                for pid in cluster_info['paper_ids']:
                    row = df.iloc[pid]
                    paper_data = {
                        "paper_id": pid,
                        "title": row['title'],
                    }
                    
                    if 'abstract' in row:
                        paper_data["abstract"] = row['abstract']
                    
                    if 'problem' in row:
                        paper_data["problem"] = row['problem']
                    if 'solution' in row:
                        paper_data["solution"] = row['solution']
                    if 'results' in row:
                        paper_data["results"] = row['results']
                    
                    file_name = row.get('file_name')
                    if file_name:
                        try:
                            file_path = Path(self.base_path) / "dataset_10k" / file_name
                            if file_path.exists():
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    paper_json = json.load(f)
                                    if isinstance(paper_json, list):
                                        for p in paper_json:
                                            if p.get('title') == row['title'] and 'semantic_scholar' in p:
                                                paper_data["semantic_scholar"] = p['semantic_scholar']
                                                break
                                    elif isinstance(paper_json, dict) and paper_json.get('title') == row['title']:
                                        if 'semantic_scholar' in paper_json:
                                            paper_data["semantic_scholar"] = paper_json['semantic_scholar']
                        except Exception as e:
                            print(f"Warning: Could not extract semantic_scholar data for {file_name}: {e}")
                    
                    papers.append(paper_data)
                node["children"] = papers
            else:
                children = []
                for ch_id in cluster_info.get('child_clusters', []):
                    children.append(build_node(level - 1, ch_id))
                node["children"] = children

            return node

        top_level_clusters = hierarchy_info['levels'][max_level]
        hierarchy = {
            "clusters": [
                build_node(max_level, cid) for cid in sorted(top_level_clusters.keys())
            ]
        }
        return hierarchy
        
    def _build_nested_json_top_down(self, hierarchy_info: Dict, max_level: int) -> Dict:
        """
        Build a nested JSON from the top level down to the papers.
        For top-down approach.
        """
        df = hierarchy_info['papers']

        def build_node(level: int, cid: int) -> Dict:
            cluster_info = hierarchy_info['levels'][level][cid]
            node = {
                "cluster_id": cid,
                "title": cluster_info['name'],
                "abstract": cluster_info['summary']
            }
            
            if level == max_level:
                papers = []
                for pid in cluster_info['paper_ids']:
                    row = df.iloc[pid]
                    paper_data = {
                        "paper_id": pid,
                        "title": row['title'],
                    }
                    
                    if 'abstract' in row:
                        paper_data["abstract"] = row['abstract']
                    
                    if 'problem' in row:
                        paper_data["problem"] = row['problem']
                    if 'solution' in row:
                        paper_data["solution"] = row['solution']
                    if 'results' in row:
                        paper_data["results"] = row['results']
                    
                    file_name = row.get('file_name')
                    if file_name:
                        try:
                            file_path = Path(self.base_path) / "dataset_10k" / file_name
                            if hasattr(self, 'input_folder') and self.input_folder:
                                file_path = Path(self.input_folder) / file_name
                                
                            if file_path.exists():
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    paper_json = json.load(f)
                                    if isinstance(paper_json, list):
                                        for p in paper_json:
                                            if p.get('title') == row['title'] and 'semantic_scholar' in p:
                                                paper_data["semantic_scholar"] = p['semantic_scholar']
                                                break
                                    elif isinstance(paper_json, dict) and paper_json.get('title') == row['title']:
                                        if 'semantic_scholar' in paper_json:
                                            paper_data["semantic_scholar"] = paper_json['semantic_scholar']
                        except Exception as e:
                            print(f"Warning: Could not extract semantic_scholar data for {file_name}: {e}")
                    
                    papers.append(paper_data)
                node["children"] = papers
            else:
                children = []
                for ch_id in cluster_info.get('child_clusters', []):
                    children.append(build_node(level + 1, ch_id))
                node["children"] = children

            return node

        top_level_clusters = hierarchy_info['levels'][1]
        hierarchy = {
            "clusters": [
                build_node(1, cid) for cid in sorted(top_level_clusters.keys())
            ]
        }
        return hierarchy

    def save_hierarchy(self, hierarchy: dict, file_path: Path) -> None:
        """
        Save hierarchy to a JSON file.
        
        Args:
            hierarchy: Hierarchy structure to save
            file_path: Output file path
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Saved hierarchy to {file_path}")