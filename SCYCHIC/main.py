"""
main.py: Orchestrates multi-run and multi-evaluate pipeline for paper clustering. 
Handles embedding generation, summarization, clustering, and evaluation with proper resource management.

Usage example:
  python main.py \
    --embedding_generator openai \
    --summary_generator llama \
    --clustering_method kmeans \
    --clustering_direction bidirectional \
    --evaluator llama \
    --base_path /path/to/project \
    --cluster_sizes 10 5 2 \
    --run_time 1 \
    --evaluate_time 3 \
    --test_count 200 \
    --pre_generated_embeddings_file /path/to/embeddings.pkl \
    --embedding_source problem
"""

import argparse
import json
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from datetime import datetime
from pathlib import Path
import numpy as np
from pipeline.pipeline import PaperProcessor
from evaluator.evaluate import get_evaluator
from dotenv import load_dotenv
import torch
import logging
import sys
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

def print_gpu_memory():
    """Print current GPU memory usage for all available devices."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2
            memory_cached = torch.cuda.memory_reserved(i) / 1024**2
            logger.info(f"GPU {i}: Allocated: {memory_allocated:.2f}MB, Cached: {memory_cached:.2f}MB")

def setup_argument_parser():
    """Configure and return the argument parser for the script."""
    parser = argparse.ArgumentParser(description="Paper clustering pipeline with multi-run support")
    
    parser.add_argument("--embedding_generator", type=str, default="openai",
                      help="Embedding generator type (e.g., openai)")
    parser.add_argument("--summary_generator", type=str, default="llama",
                      help="Summary generator type (e.g., llama)")
    parser.add_argument("--clustering_method", type=str, default="kmeans",
                      help="Clustering method (e.g., kmeans)")
    parser.add_argument("--clustering_direction", type=str, default="bottom_up",
                      help="Direction of clustering ('bottom_up', 'top_down', or 'bidirectional')")
    parser.add_argument("--evaluator", type=str, default="llama",
                      help="Evaluator type (e.g., llama, mistral, gpt4)")
    parser.add_argument("--evaluate_type", type=str, default="normal",
                    help="Evaluation approach: 'normal' (direct matching) or 'question' (search query)")

    parser.add_argument("--base_path", type=str, default="./my_project_data",
                      help="Base directory for input_papers, results, etc.")
    parser.add_argument("--cluster_sizes", nargs='+', type=int, default=[438, 78, 17],
                      help="Multi-level cluster sizes (e.g., 438 78 17)")
    parser.add_argument("--run_time", type=int, default=5,
                      help="Number of pipeline runs")
    parser.add_argument("--evaluate_time", type=int, default=3,
                      help="Number of evaluations per run")
    parser.add_argument("--test_count", type=int, default=200,
                      help="Number of test samples for evaluation")

    parser.add_argument("--openai_key", type=str, default=os.getenv('OPENAI_API_KEY'),
                      help="OpenAI API key (if not in environment)")
    parser.add_argument("--cluster_prompt_folder", type=str, default="./prompts/",
                      help="Prompt template file for cluster naming")
    parser.add_argument("--input_folder", type=str, default=None,
                      help="Custom input papers folder path")

    parser.add_argument("--pre_generated_embeddings_file", type=str, default='/weka/scratch/dkhasha1/mgao38/taxonomy/embeddings/key_embeddings.pkl',
                      help="Path to pre-generated embeddings file (.pkl)")
    parser.add_argument("--embedding_source", type=str, default="all",
                      help="Source of embeddings to use: 'all' for all embeddings, a category name like 'problem' "
                           "for all keys in that category, or a specific key like 'problem.overarching problem domain'")

    return parser

def convert_to_native_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in list(obj)]
    return obj

def determine_embedding_type_and_key(embedding_source):
    """
    Determine embedding type and key based on the embedding_source format
    
    Returns:
        tuple: (embedding_type, embedding_key)
    """
    if embedding_source == "all":
        return "all", None
    elif "." in embedding_source:
        return "subkey", embedding_source
    else:
        return "key", embedding_source

def get_prompt_file_for_embedding_type(embedding_type, embedding_key, base_prompt_folder):
    """
    Select the appropriate prompt file based on embedding type and key.
    
    Args:
        embedding_type: Type of embedding ('all', 'key', or 'subkey')
        embedding_key: The specific key being used (if applicable)
        base_prompt_folder: Base folder for prompt templates
    
    Returns:
        str: Path to the appropriate prompt file
    """
    base_folder = Path(base_prompt_folder)
    
    default_prompt = base_folder / "default_prompt.txt"
    
    if embedding_type == "all":
        return str(base_folder / "all_embeddings_prompt.txt")
    elif embedding_type == "key":
        if embedding_key == "problem":
            return str(base_folder / "problem_prompt.txt")
        elif embedding_key == "solution":
            return str(base_folder / "solution_prompt.txt")
        elif embedding_key == "results":
            return str(base_folder / "results_prompt.txt")
    elif embedding_type == "subkey":
        if embedding_key.startswith("problem."):
            return str(base_folder / "problem_subkey_prompt.txt")
        elif embedding_key.startswith("solution."):
            return str(base_folder / "solution_subkey_prompt.txt")
    
    logger.info(f"No specific prompt file found for {embedding_type}/{embedding_key}, using default")
    return str(default_prompt)

def process_single_run(run_idx, args, now_str, size_str, evaluator_instance):
    """Execute a single run of the pipeline with proper resource management."""
    processor = None
    try:
        logger.info(f"\n=== Starting Run {run_idx}/{args.run_time} ===")
        print_gpu_memory()
        
        seed = 1000 + run_idx * 37
        logger.info(f"Using seed: {seed}")
        
        embedding_type, embedding_key = determine_embedding_type_and_key(args.embedding_source)
        logger.info(f"Using embedding source: {args.embedding_source} (Type: {embedding_type}, Key: {embedding_key if embedding_key else 'None'})")
        
        embedding_info = ""
        if args.pre_generated_embeddings_file:
            safe_source = args.embedding_source.replace(".", "-")
            embedding_info = f"_emb_{safe_source}"

        prompt_file = get_prompt_file_for_embedding_type(
            embedding_type, 
            embedding_key, 
            args.cluster_prompt_folder
        )
        logger.info(f"Using prompt file: {prompt_file}")
        
        direction_info = f"_{args.clustering_direction}"
        sub_name = f"{now_str}_{args.embedding_generator}_{args.summary_generator}_{args.clustering_method}{embedding_info}{direction_info}_{size_str}_{seed}"

        processor = PaperProcessor(
            base_path=args.base_path,
            embedding_generator=args.embedding_generator,
            summary_generator=args.summary_generator,
            clustering_method=args.clustering_method,
            clustering_direction=args.clustering_direction,
            openai_key=args.openai_key,
            cluster_prompt_file=prompt_file,
            random_seed=seed,
            embeddings_subfolder=sub_name,
            embedding_type=embedding_type,
            embedding_key=embedding_key,
            pre_generated_embeddings_file=args.pre_generated_embeddings_file
        )

        logger.info(f"Starting hierarchical clustering (direction: {args.clustering_direction})...")
        final_hierarchy = processor.process_hierarchical_clustering(
            cluster_sizes=args.cluster_sizes,
            run_seed=seed,
            input_folder=args.input_folder,
        )

        logger.info(f"Hierarchy structure generated: {final_hierarchy is not None}")
        if final_hierarchy:
            logger.info(f"Top-level keys: {list(final_hierarchy.keys())}")
            if 'clusters' in final_hierarchy:
                logger.info(f"Number of top clusters: {len(final_hierarchy['clusters'])}")

        out_json_name = f"hierarchy_{sub_name}.json"
        out_json_path = Path(args.base_path) / "results" / out_json_name
        
        final_hierarchy = convert_to_native_types(final_hierarchy)
        with open(out_json_path, 'w', encoding='utf-8') as fout:
            json.dump(final_hierarchy, fout, indent=2, ensure_ascii=False)
        logger.info(f"Saved hierarchy to: {out_json_path}")

        logger.info("\n=== After Processing ===")
        print_gpu_memory()
        
        if processor:
            logger.info("Cleaning up processor...")
            processor.cleanup()

        for eval_idx in range(1, args.evaluate_time + 1):
            logger.info(f"\n=== Starting Evaluation {eval_idx}/{args.evaluate_time} for Run {run_idx} ===")
            evaluator_instance.evaluate_hierarchy(
                hierarchy_path=str(out_json_path),
                abstracts_folder=str(Path(args.base_path)/"dataset_10k") if not args.input_folder else args.input_folder,
                test_count=args.test_count,
                output_file=str(Path(args.base_path)/"results"/"evaluation_results.csv"),
                embedding_key=args.embedding_source
            )

    except Exception as e:
        logger.error(f"Error in run {run_idx}: {str(e)}", exc_info=True)
        raise e

    finally:
        logger.info(f"\n=== Final Cleanup for Run {run_idx} ===")
        if processor is not None:
            logger.info("Performing final cleanup of processor...")
            try:
                processor.cleanup()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        
        logger.info("Running garbage collection...")
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Final cleanup complete")
        print_gpu_memory()

def main():
    """Main execution function for the pipeline."""
    try:
        print(os.environ.get("SPAWNED_PROCESS"))
        multiprocessing.set_start_method('spawn', force=True)
        print(os.environ.get("SPAWNED_PROCESS"))
        parser = setup_argument_parser()
        args = parser.parse_args()
        
        if args.pre_generated_embeddings_file:
            if not os.path.exists(args.pre_generated_embeddings_file):
                logger.error(f"Pre-generated embeddings file not found: {args.pre_generated_embeddings_file}")
                return
                
            logger.info(f"Using pre-generated embeddings from {args.pre_generated_embeddings_file}")
            logger.info(f"Embedding source: {args.embedding_source}")
            
        if args.clustering_direction not in ["bottom_up", "top_down", "bidirectional"]:
            logger.error(f"Invalid clustering direction: {args.clustering_direction}. Must be 'bottom_up', 'top_down', or 'bidirectional'.")
            return
        
        logger.info(f"Clustering direction: {args.clustering_direction}")

        now_str = datetime.now().strftime("%Y%m%d")
        size_str = "-".join(map(str, args.cluster_sizes))
        
        logger.info("Initializing evaluator...")
        evaluator_instance = get_evaluator(
            args.evaluator, 
            embedding_key=args.embedding_source,
            evaluate_type=args.evaluate_type
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        print(f"The cluster size is set as {args.cluster_sizes}", flush=True)
        for run_idx in range(1, args.run_time + 1):
            process_single_run(run_idx, args, now_str, size_str, evaluator_instance)

    except Exception as e:
        logger.error("Fatal error in main execution", exc_info=True)
        raise e

    finally:
        logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()