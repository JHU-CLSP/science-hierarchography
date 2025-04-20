import os
import re
import json
import copy
import unicodedata
from unidecode import unidecode
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import argparse
import concurrent.futures
torch.cuda.empty_cache()
import gc
gc.collect()

# Qwen imports
from transformers import AutoModelForCausalLM, AutoTokenizer


###############################################################################
# 1) Taxonomy Processor
###############################################################################
class TaxonomyProcessor:
    def __init__(self, taxonomy_json):
        self.taxonomy = taxonomy_json
        self.paper_to_categories = {}
        self._process_taxonomy(self.taxonomy, [])

    def _normalize_title(self, title):
        normalized = unicodedata.normalize('NFKC', title)
        transliterated = unidecode(normalized)
        return transliterated.casefold().strip()

    def _process_taxonomy(self, node, current_path):
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = current_path + [key]
                if isinstance(value, dict) and "Papers" in value:
                    papers = value["Papers"]
                    if isinstance(papers, list):
                        for paper in papers:
                            raw_title = paper.get("Title")
                            if raw_title:
                                norm_title = self._normalize_title(raw_title)
                                if norm_title not in self.paper_to_categories:
                                    self.paper_to_categories[norm_title] = []
                                self.paper_to_categories[norm_title].append(new_path)
                self._process_taxonomy(value, new_path)
        elif isinstance(node, list):
            for item in node:
                self._process_taxonomy(item, current_path)

    def get_paper_categories(self, paper_title):
        norm_title = self._normalize_title(paper_title)
        return self.paper_to_categories.get(norm_title)


###############################################################################
# 2) Base Evaluator
###############################################################################
class BaseEvaluator:
    def evaluate_hierarchy(self, hierarchy_path: str, abstracts_folder: str, test_count: int, output_file: str, embedding_key: str = None):
        raise NotImplementedError("Subclasses must implement evaluate_hierarchy.")


###############################################################################
# 3) Qwen Evaluator
###############################################################################
class QwenEvaluator(BaseEvaluator):
    def __init__(self, model_name="Qwen/Qwen2.5-32B-Instruct"):
        self.model_name = model_name
        print(f"[Evaluator] Using {model_name} locally with Transformers for classification.")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True
        )

    def _ask_qwen_to_choose_category(self, paper_title, paper_content, path_so_far, candidate_categories):
        prompt = f"""You are a scientist expert in taxonomy. Please read the following paper title and abstract.
        Your task is to choose the next topic (while considering the current path) in the taxonomy that has the best chance of containing this paper.
Paper:
  Title: {paper_title}
  Abstract (snippet): {paper_content[:300]}...
Current Path: {path_so_far or "Root"}
Choose from this topics list (MUST pick one):
topics = {', '.join(candidate_categories)}

Required Response Format:
Topic: [EXACT topic name from the "topics"]
Compulsorily choose one topic from the existing list and do not make up any new topics
"""
        messages = [
            {"role": "system", "content": "You are a scientist experienced at reading scientific papers."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=50)
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # print(f"[Qwen] Qwen's response: {response}")
        # Extract 'Topic: [name]' line
        # match = re.search(r"Topic:\s*(.*?)\n", response, re.DOTALL)
        # if match:
        #     chosen_category = match.group(1).strip()
        # else:
        #     chosen_category = response.split("\n")[0].replace("Topic:", "").strip()
        matches = re.findall(r"Topic:\s*(\S.+)", response)

        if len(matches) >= 2:
            # Return the second occurrence
            chosen_category = matches[1].strip()
        elif matches:
            # Fallback if only one match is found
            chosen_category = matches[0].strip()
        else:
            # If 'Topic:' is not found at all
            chosen_category = response.split("\n")[0].replace("Topic:", "").strip()

        if chosen_category not in candidate_categories:
            print(f"[ERROR] Qwen's choice '{chosen_category}' is not among candidates: {candidate_categories}",flush=True)
            return "Invalid Category"

        return chosen_category


###############################################################################
# 4) Evaluate Single Paper
###############################################################################
def evaluate_single_paper(evaluator, taxonomy_path, paper_title, paper_abstract):
    with open(taxonomy_path, "r", encoding="utf-8") as file:
        hierarchy_data = json.load(file)

    processor = TaxonomyProcessor(hierarchy_data)
    top_level_categories = list(hierarchy_data.keys())
    true_paths = processor.get_paper_categories(paper_title)

    if not true_paths:
        return {
            "paper_title": paper_title,
            "true_paths": [],
            "predicted_path": "",
            "correct": False,
            "error": "No category found in taxonomy."
        }

    predicted_path = []
    hierarchy_copy = copy.deepcopy(hierarchy_data)
    current_candidates = top_level_categories

    while current_candidates:
        chosen_category = evaluator._ask_qwen_to_choose_category(
            paper_title, paper_abstract, predicted_path, current_candidates
        )
        if chosen_category not in hierarchy_copy:
            predicted_path.append("Invalid Category")
            break

        predicted_path.append(chosen_category)
        hierarchy_copy = hierarchy_copy[chosen_category]

        if isinstance(hierarchy_copy, dict) and "Papers" in hierarchy_copy:
            break

        if isinstance(hierarchy_copy, dict):
            current_candidates = [k for k in hierarchy_copy.keys() if k != "Papers"]
        else:
            break

    is_correct = False
    if predicted_path and predicted_path[-1] != "Invalid Category":
        predicted_last = predicted_path[-1]
        for pth in true_paths:
            if pth and (pth[-1] == predicted_last):
                is_correct = True
                break

    return {
        "paper_title": paper_title,
        "true_paths": [" > ".join(path) for path in true_paths],
        "predicted_path": " > ".join(predicted_path) if predicted_path else "",
        "correct": is_correct,
    }


###############################################################################
# 5) Evaluate Multiple Papers
###############################################################################
def evaluate_multiple_papers(evaluator, taxonomy_path, csv_path, sample_size=200, max_workers=2):
    with open(taxonomy_path, "r", encoding="utf-8") as file:
        hierarchy_data = json.load(file)

    processor = TaxonomyProcessor(hierarchy_data)
    df = pd.read_csv(csv_path)

    if "Title" not in df.columns or "Abstract" not in df.columns:
        raise ValueError("CSV must contain 'Title' and 'Abstract' columns")

    sampled_papers = df.sample(n=sample_size, random_state=30)
    results = []
    fully_correct_count = 0
    total_node_matches = 0.0
    total_nodes_predicted = 0.0
    total_nodes_actual_sum = 0.0
    valid_evaluations = 0

    futures_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in sampled_papers.iterrows():
            future = executor.submit(
                evaluate_single_paper,
                evaluator,
                taxonomy_path,
                row["Title"],
                row["Abstract"]
            )
            futures_dict[future] = (idx, row)

        for future in tqdm(concurrent.futures.as_completed(futures_dict), total=len(futures_dict), desc="Evaluating"):
            idx, row = futures_dict[future]
            result = future.result()

            if "error" in result:
                results.append({
                    "paper_title": row["Title"],
                    "paper_abstract": row["Abstract"],
                    "true_paths": "None",
                    "predicted_path": "",
                    "correct": False,
                    "error": result["error"],
                    "matched_nodes": 0,
                    "predicted_length": 0,
                })
                continue

            valid_evaluations += 1
            # results.append({
            #     "paper_title": row["Title"],
            #     "paper_abstract": row["Abstract"],
            #     "true_paths": " | ".join(result["true_paths"]),
            #     "predicted_path": result["predicted_path"],
            #     "correct": result["correct"],
            # })

            if result["correct"]:
                fully_correct_count += 1

            predicted_nodes = result["predicted_path"].split(" > ") if result["predicted_path"] else []
            true_paths = [tp.split(" > ") for tp in result["true_paths"]] if result["true_paths"] else []

            avg_true_path_length = sum(len(tp) for tp in true_paths) / len(true_paths) if true_paths else 0
            total_nodes_actual_sum += avg_true_path_length
            total_nodes_predicted += len(predicted_nodes)

            best_match_for_this_paper = 0
            best_levelwise_match = []

            for tpath in true_paths:
                match_count = 0
                levelwise = []
                for i in range(max(len(predicted_nodes), len(tpath))):
                    pred = predicted_nodes[i] if i < len(predicted_nodes) else None
                    true = tpath[i] if i < len(tpath) else None
                    is_match = int(pred == true and pred is not None)
                    levelwise.append(is_match)
                    if is_match:
                        match_count += 1
                    else:
                        break  # Stop at first mismatch (ordered match)

                if match_count > best_match_for_this_paper:
                    best_match_for_this_paper = match_count
                    best_levelwise_match = levelwise

            total_node_matches += best_match_for_this_paper

            # Pad levelwise match to a fixed number of levels (optional)
            max_depth = 10  # Set a safe limit
            padded_levelwise = best_levelwise_match + [0] * (max_depth - len(best_levelwise_match))

            # Build row result
            row_result = {
                "paper_title": row["Title"],
                "paper_abstract": row["Abstract"],
                "true_paths": " | ".join(result["true_paths"]),
                "predicted_path": result["predicted_path"],
                "correct": result["correct"],
                "matched_nodes": best_match_for_this_paper,
                "true_path_length": avg_true_path_length,
                "predicted_path_length": len(predicted_nodes),
            }
            for i in range(max_depth):
                row_result[f"Level_{i+1}_Match"] = padded_levelwise[i]

            results.append(row_result)


    exact_match_accuracy = fully_correct_count / valid_evaluations if valid_evaluations > 0 else 0.0
    avg_node_match = total_node_matches / total_nodes_actual_sum if total_nodes_actual_sum else 0
    avg_predicted_length = total_nodes_predicted / valid_evaluations if valid_evaluations else 0
    levelwise_columns = [f"Level_{i+1}_Match" for i in range(10)]
    levelwise_accuracies = {}

    for col in levelwise_columns:
        if any(col in r for r in results):
            col_values = [r[col] for r in results if col in r]
            if col_values:
                levelwise_accuracies[col] = sum(col_values) / len(col_values)

    print("\n=== Evaluation Summary ===",flush=True)
    print(f"Total Papers: {sample_size}",flush=True)
    print(f"Valid Papers: {valid_evaluations}",flush=True)
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}",flush=True)
    print(f"Avg Node Match: {avg_node_match:.2%}",flush=True)
    print(f"Avg Predicted Path Length: {avg_predicted_length:.2f}",flush=True)
    print("\n=== Levelwise Accuracy ===",flush=True)
    for level, acc in levelwise_accuracies.items():
        print(f"{level}: {acc:.2%}")
    print("-" * 80,flush=True)

    return pd.DataFrame(results)

###############################################################################
# 6) Run Example
###############################################################################
def main():
    parser = argparse.ArgumentParser(description="Evaluate scientific hierarchy using Qwen model.")
    parser.add_argument("--taxonomy", required=True, help="Path to taxonomy JSON file")
    parser.add_argument("--csv", required=True, help="Path to input CSV file with 'Title' and 'Abstract'")
    parser.add_argument("--output", required=True, help="Path to output Excel file")
    parser.add_argument("--sample_size", type=int, default=200, help="Number of papers to evaluate (default: 200)")
    parser.add_argument("--max_workers", type=int, default=2, help="Number of threads for parallel evaluation")

    args = parser.parse_args()

    evaluator = QwenEvaluator()
    results_df = evaluate_multiple_papers(
        evaluator,
        taxonomy_path=args.taxonomy,
        csv_path=args.csv,
        sample_size=args.sample_size,
        max_workers=args.max_workers
    )

    results_df.to_excel(args.output, index=False)
    print(f"[✔] Results saved to {args.output}", flush=True)
    
if __name__ == "__main__":
    main()
    # taxonomy_file = "/weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/llama/code/evaluation/results/final_llm_incr_leaf.json"
    # csv_file = "/weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/clean_dataset_final.csv"
    # output_file = "/weka/scratch/dkhasha1/jash09/science-cartography-1/topic_clustering/llama/code/evaluation/results/qwen_results_llm_incr_2.xlsx"

    # results_df = evaluate_multiple_papers(
    #     qwen_evaluator,
    #     taxonomy_file,
    #     csv_file,
    #     sample_size=200
    # )
    # results_df.to_excel(output_file, index=False)
    # print(f"Saved results to {output_file}",flush=True)
    

