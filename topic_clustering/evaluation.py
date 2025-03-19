import os
import re
import json
import csv
import copy
import random
import openai
import unicodedata
from unidecode import unidecode
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent.futures
# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key

###############################################################################
# 1) Taxonomy Processor
###############################################################################
class TaxonomyProcessor:
    """
    Reads a JSON taxonomy and extracts the path(s) at which each paper title is found.
    """

    def __init__(self, taxonomy_json):
        self.taxonomy = taxonomy_json
        self.paper_to_categories = {}
        self._process_taxonomy(self.taxonomy, [])

    def _normalize_title(self, title):
        """Normalize titles to handle Unicode variations, case sensitivity, and transliteration."""
        # Normalize to NFKC form
        normalized = unicodedata.normalize('NFKC', title)
        # Transliterate to ASCII
        transliterated = unidecode(normalized)
        # Case fold for case-insensitive comparison
        return transliterated.casefold().strip()

    def _process_taxonomy(self, node, current_path):
        # Recursively traverse the taxonomy to map each paper title to its path.
        if isinstance(node, dict):
            for key, value in node.items():
                new_path = current_path + [key]

                # Process "Papers" if present in this node
                if isinstance(value, dict) and "Papers" in value:
                    papers = value["Papers"]
                    if isinstance(papers, list):
                        for paper in papers:
                            raw_title = paper.get("Title")  # ensure we have a "Title" key
                            if raw_title:
                                norm_title = self._normalize_title(raw_title)
                                if norm_title not in self.paper_to_categories:
                                    self.paper_to_categories[norm_title] = []
                                self.paper_to_categories[norm_title].append(new_path)

                # Recursively process nested structures (dicts or lists)
                self._process_taxonomy(value, new_path)

        elif isinstance(node, list):
            for item in node:
                self._process_taxonomy(item, current_path)

    def get_paper_categories(self, paper_title):
        """
        Returns a list of all possible taxonomy paths (list of strings)
        for the given paper title. Returns None if not found.
        """
        norm_title = self._normalize_title(paper_title)
        return self.paper_to_categories.get(norm_title)


###############################################################################
# 2) Base Evaluator (abstract)
###############################################################################
class BaseEvaluator:
    """
    Abstract base class for different evaluator implementations.
    """
    def evaluate_hierarchy(self, hierarchy_path: str, abstracts_folder: str, test_count: int, output_file: str, embedding_key: str = None):
        raise NotImplementedError("Subclasses must implement evaluate_hierarchy.")


###############################################################################
# 3) GPT-4 Evaluator
###############################################################################
class GPT4Evaluator(BaseEvaluator):
    """
    Uses OpenAI's GPT-4 API (or GPT-3.5, or any other model) to classify
    papers into a hierarchical taxonomy level-by-level.
    """
    def __init__(self, model_name="gpt-4", api_key=None, embedding_key=None):
        self.model_name = model_name
        self.embedding_key = embedding_key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API key for OpenAI is missing!")
        openai.api_key = self.api_key

        print(f"[Evaluator] Using OpenAI {model_name} model for classification.")

    def evaluate_hierarchy(self, hierarchy_path: str, abstracts_folder: str, test_count: int, output_file: str, embedding_key: str = None):
        """
        Example method that takes a folder of paper files (abstracts) and a taxonomy JSON,
        classifies a specified number of random papers, and writes results to a CSV.
        """
        if embedding_key is not None:
            self.embedding_key = embedding_key

        # Load hierarchy
        with open(hierarchy_path, "r", encoding="utf-8") as file:
            hierarchy_data = json.load(file)

        # Extract categories and mappings
        processor = TaxonomyProcessor(hierarchy_data)
        top_level_categories = list(hierarchy_data.keys())

        # Read paper files
        abs_path = Path(abstracts_folder)
        all_files = [f for f in abs_path.iterdir() if f.suffix in [".txt", ".json"]]
        random.shuffle(all_files)
        test_files = all_files[:test_count]

        # Prepare tracking
        level_correct = {}
        level_total = {}

        # Determine maximum depth from your taxonomy (if needed)
        max_depth = max(len(path) for path in processor.paper_to_categories.values() if path)
        for lv in range(max_depth):
            level_correct[lv] = 0
            level_total[lv] = 0

        results = []

        for test_file in tqdm(test_files, desc="Evaluating papers"):
            paper_title, paper_content = self._parse_file(test_file)
            
            # Get the ground-truth path(s)
            true_paths = processor.get_paper_categories(paper_title)
            if not true_paths:
                # Skip if no category is found
                print(f"[WARNING] No category found for {paper_title}. Skipping...")
                continue

            # Evaluate with GPT (stops on invalid pick)
            predicted_path = self._evaluate_paper_hierarchy(
                paper_title, 
                paper_content, 
                hierarchy_data, 
                top_level_categories
            )

            # Take just the first ground-truth path (or handle multiple paths if you prefer).
            true_path = true_paths[0]

            # Compare predicted vs. true path level-by-level
            # We'll compare only as many levels as GPT actually predicted
            # so if GPT stopped early, we won't evaluate beyond that.
            max_compare_levels = min(len(true_path), len(predicted_path))
            for lv in range(max_compare_levels):
                level_total[lv] += 1
                if predicted_path[lv] == true_path[lv]:
                    level_correct[lv] += 1

            # Determine correctness at the final node (only if we have predicted at least that many levels)
            is_correct = False
            if len(predicted_path) == len(true_path):
                if predicted_path and (predicted_path[-1] == true_path[-1]):
                    is_correct = True

            results.append({
                "paper_title": paper_title,
                "true_path": " > ".join(true_path),
                "predicted_path": " > ".join(predicted_path),
                "correct": is_correct
            })

        # Print level-wise accuracy
        print("\n=== Level-Wise Accuracy ===")
        for lv in sorted(level_total.keys()):
            total = level_total[lv]
            correct = level_correct[lv]
            accuracy = correct / total if total > 0 else 0
            print(f"Level {lv} Accuracy: {accuracy:.2%} ({correct}/{total})")


        # Write results to CSV
        self._write_results_csv(results, output_file, level_correct, level_total)

    def _evaluate_paper_hierarchy(self, paper_title, paper_content, hierarchy_data, top_level_categories):
        """
        Iterates through the taxonomy level by level.
        If GPT picks an invalid category, we stop immediately
        and return whatever partial path we have so far.
        """
        current_candidates = top_level_categories
        chosen_path = []
        current_node = hierarchy_data

        while current_candidates:
            chosen_category = self._ask_gpt4_to_choose_category(
                paper_title,
                paper_content,
                chosen_path,
                current_candidates
            )

            if chosen_category not in current_node:
                print(f"[INFO] Stopping deeper evaluation for {paper_title}: GPT chose invalid '{chosen_category}'.")
                # We do NOT append 'Invalid Category' to the path in this approach.
                # We just break here, so we keep the partial valid path only.
                break

            chosen_path.append(chosen_category)
            current_node = current_node[chosen_category]

            # If we see "Papers", we've reached a leaf node. Stop.
            if isinstance(current_node, dict) and "Papers" in current_node:
                break

            # Gather next-level candidates
            if isinstance(current_node, dict):
                current_candidates = [k for k in current_node.keys() if k != "Papers"]
            else:
                break

        return chosen_path

    def _ask_gpt4_to_choose_category(self, paper_title, paper_content, path_so_far, candidate_categories):
        """
        Prompts GPT-4 to pick a category from candidate_categories.
        Must return one EXACT string from candidate_categories.
        """
        context = f"""You are a scientist expert in taxonomy. Please read the following paper title and abstract, pick one topic from the list of topics best matches the paper's content.
Paper:
  Title: {paper_title}
  Abstract (snippet): {paper_content[:300]}...
Current Path: {path_so_far or "Root"}
Choose from this topics list (MUST pick one):
{', '.join(candidate_categories)}

Required Response Format:
Topic: [EXACT category name from the list]
"""

        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a scientist expert in scientific taxonomy selection."},
                {"role": "user", "content": context}
            ],
            temperature=0.0,
            max_tokens=100
        )

        response_text = response["choices"][0]["message"]["content"].strip()

        # Attempt to parse "Topic:" line
        match = re.search(r"Topic:\s*(.*?)\n", response_text, re.DOTALL)
        if match:
            chosen_category = match.group(1).strip()
        else:
            # Fallback: parse the first line after "Topic:"
            chosen_category = response_text.split("\n")[0].replace("Topic:", "").strip()

        if chosen_category not in candidate_categories:
            print(f"[ERROR] GPT's choice '{chosen_category}' is not among candidates: {candidate_categories}")
            return "Invalid Category"

        return chosen_category

    def _parse_file(self, file_path: Path):
        """
        Example method to parse a .txt or .json file for title/abstract content.
        Adapt as needed.
        """
        paper_title = file_path.stem
        paper_content = ""
        if file_path.suffix == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                paper_content = f.read()
        elif file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                paper_content = data.get("abstract", "")
        return paper_title, paper_content

    def _write_results_csv(self, results, output_file, level_correct, level_total):
        """
        Writes the classification results & level-wise accuracies to a CSV file.
        """
        file_exists = os.path.exists(output_file)
        fieldnames = ["paper_title", "true_path", "predicted_path", "correct"]

        # Add level-wise accuracy columns
        for lv in sorted(level_total.keys()):
            fieldnames.append(f"level_{lv}_accuracy")

        with open(output_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            for result in results:
                row = result.copy()
                # For each level, repeat the final accuracy (the same for all rows)
                for lv in sorted(level_total.keys()):
                    total = level_total[lv]
                    correct = level_correct[lv]
                    accuracy = correct / total if total > 0 else 0
                    row[f"level_{lv}_accuracy"] = f"{accuracy:.4f}"
                writer.writerow(row)

###############################################################################
# 4) Single-paper evaluation
###############################################################################
def evaluate_single_paper(evaluator, taxonomy_path, paper_title, paper_abstract):
    """
    Evaluates a single paper's classification using the GPT4Evaluator pipeline.
    If no category path is found in the taxonomy, returns an error in the result.
    """
    # Load taxonomy
    with open(taxonomy_path, "r", encoding="utf-8") as file:
        hierarchy_data = json.load(file)

    # Prepare taxonomy processor
    processor = TaxonomyProcessor(hierarchy_data)
    top_level_categories = list(hierarchy_data.keys())

    # Get paper's ground-truth path(s)
    true_paths = processor.get_paper_categories(paper_title)
    if not true_paths:
        # If no path found, return error
        return {
            "paper_title": paper_title,
            "true_paths": [],
            "predicted_path": "",
            "correct": False,
            "error": "No category found in taxonomy."
        }

    # Traverse the hierarchy with GPT
    predicted_path = []
    hierarchy_copy = copy.deepcopy(hierarchy_data)
    current_candidates = top_level_categories

    while current_candidates:
        chosen_category = evaluator._ask_gpt4_to_choose_category(
            paper_title, paper_abstract, predicted_path, current_candidates
        )
        if chosen_category not in hierarchy_copy:
            # Invalid choice, break early
            predicted_path.append("Invalid Category")
            break

        predicted_path.append(chosen_category)
        hierarchy_copy = hierarchy_copy[chosen_category]

        # Stop if we reach leaf node that has "Papers"
        if isinstance(hierarchy_copy, dict) and "Papers" in hierarchy_copy:
            break

        # Next candidates (exclude "Papers")
        if isinstance(hierarchy_copy, dict):
            current_candidates = [k for k in hierarchy_copy.keys() if k != "Papers"]
        else:
            break

    # Final correctness check (compare final node to any of the known ground-truth final nodes)
    is_correct = False
    if predicted_path and predicted_path[-1] != "Invalid Category":
        predicted_last = predicted_path[-1]
        # If any of the true paths share the same final node, we consider it correct.
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
# 5) Multiple-paper evaluation
###############################################################################
def evaluate_multiple_papers(
    evaluator, 
    taxonomy_path, 
    csv_path, 
    sample_size=200,
    max_workers=4
):
    """
    Evaluates multiple papers from a CSV file that has "Title" and "Abstract" columns.
    Randomly samples `sample_size` rows, classifies them, and calculates accuracy metrics.

    - Uses a ThreadPoolExecutor for parallel calls to `evaluate_single_paper`.
    - Skips papers that do not exist in the taxonomy (no path found). They are not
      counted in the final accuracy calculations, but their results are still stored.
    - max_workers sets how many concurrent threads you want to use.
    """
    # Load taxonomy
    with open(taxonomy_path, "r", encoding="utf-8") as file:
        hierarchy_data = json.load(file)

    processor = TaxonomyProcessor(hierarchy_data)

    # Load papers from CSV
    df = pd.read_excel(csv_path)
    if "Title" not in df.columns or "Abstract" not in df.columns:
        raise ValueError("CSV file must contain 'Title' and 'Abstract' columns")

    # Sample
    sampled_papers = df.sample(n=sample_size, random_state=30)
    
    results = []
    fully_correct_count = 0

    # For node-level accuracy
    total_node_matches = 0.0
    total_nodes_predicted = 0.0
    total_nodes_actual_sum = 0.0
    valid_evaluations = 0  # Count how many we actually evaluate (have a path in the taxonomy)

    # We store the futures in a dictionary so we can retrieve (idx, row)
    # once the future completes.
    futures_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, row in sampled_papers.iterrows():
            paper_title = row["Title"]
            paper_abstract = row["Abstract"]
            future = executor.submit(
                evaluate_single_paper,
                evaluator,
                taxonomy_path,
                paper_title,
                paper_abstract
            )
            # Map this future back to (idx, row)
            futures_dict[future] = (idx, row)

        # Now we collect results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures_dict), 
            total=len(futures_dict), 
            desc="Evaluating multiple papers"
        ):
            idx, row = futures_dict[future]
            # Get the result (or re-raise any exception that happened in the thread)
            result = future.result()  

            # If there's an error (no path found, etc.), record it but don't affect accuracy
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

            # Now we know the paper had at least one valid path
            valid_evaluations += 1
            results.append({
                "paper_title": row["Title"],
                "paper_abstract": row["Abstract"],
                "true_paths": " | ".join(result["true_paths"]),
                "predicted_path": result["predicted_path"],
                "correct": result["correct"],
            })

            if result["correct"]:
                fully_correct_count += 1

            # Compute node-level matches
            predicted_nodes = result["predicted_path"].split(" > ") if result["predicted_path"] else []
            true_paths = [tp.split(" > ") for tp in result["true_paths"]] if result["true_paths"] else []

            # Find best node overlap among multiple possible true paths
            best_match_for_this_paper = 0
            for tpath in true_paths:
                match_count = 0
                for i in range(min(len(predicted_nodes), len(tpath))):
                    if predicted_nodes[i] == tpath[i]:
                        match_count += 1
                    else:
                        break
                if match_count > best_match_for_this_paper:
                    best_match_for_this_paper = match_count

            total_node_matches += best_match_for_this_paper
            total_nodes_predicted += len(predicted_nodes)

            # Average length among all the paper’s true paths
            if true_paths:
                avg_length = sum(len(tp) for tp in true_paths) / len(true_paths)
            else:
                avg_length = 0
            total_nodes_actual_sum += avg_length

    # Final metrics (only for those with a valid taxonomy path)
    exact_match_accuracy = 0.0
    avg_node_match = 0.0
    avg_predicted_length = 0.0
    if valid_evaluations > 0:
        exact_match_accuracy = fully_correct_count / valid_evaluations
        avg_node_match = (
            total_node_matches / total_nodes_actual_sum 
            if total_nodes_actual_sum else 0
        )
        avg_predicted_length = total_nodes_predicted / valid_evaluations

    print("\n=== Overall Evaluation Metrics (for valid papers only) ===")
    print(f"Total Papers Sampled: {sample_size}")
    print(f"Valid Papers (found in taxonomy): {valid_evaluations}")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.2%} ({fully_correct_count}/{valid_evaluations})")
    print(f"Average Node Match (percentage): {avg_node_match:.2%}")
    print(f"Average Predicted Path Length: {avg_predicted_length:.2f}")
    print("-" * 80)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    return results_df



###############################################################################
# EXAMPLE USAGE (commented out)
###############################################################################

if __name__ == "__main__":
    
    # 1) Initialize the evaluator
    gpt_evaluator = GPT4Evaluator(api_key=openai_api_key)

    # 2) Evaluate a single paper
    taxonomy_file = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/results/mapped_taxonomy.json"
    # single_title = "Some Paper Title"
    # single_abstract = "Some Abstract text..."
    # single_result = evaluate_single_paper(gpt_evaluator, taxonomy_file, single_title, single_abstract)
    # print("Single-paper result:", single_result)

    # 3) Evaluate multiple papers from a CSV
    csv_file = "/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/sampled_papers.xlsx"
    results_df = evaluate_multiple_papers(gpt_evaluator, taxonomy_file, csv_file, sample_size=25)
    results_df.to_excel("/Users/jashshah/Desktop/science-cartography-1/topic_clustering/taxonomies/exp/taxo_deepen/evaluation_results_bef_deep.xlsx", index=False)
    print("Saved multiple-paper evaluation results to evaluation_output.xlsx")

