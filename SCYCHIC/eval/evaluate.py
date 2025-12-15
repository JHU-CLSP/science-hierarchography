import os
import random
import json
import re
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

import transformers
import torch
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class BaseEvaluator:
    def evaluate_hierarchy(self, hierarchy_path: str, abstracts_folder: str, test_count: int,
                          output_file: str, embedding_key: str = None, query_type: str = None, run_number: int = 1):
        raise NotImplementedError("Subclasses must implement evaluate_hierarchy.")


def get_evaluator(name: str, embedding_key: str = None, evaluate_type: str = "normal", query_type: str = "random"):
    name = name.lower()

    if evaluate_type not in ["normal", "question"]:
        raise ValueError(f"evaluate_type must be 'normal' or 'question', got '{evaluate_type}'")

    if name == "llama":
        return LlamaEvaluator(embedding_key=embedding_key, evaluate_type=evaluate_type, query_type=query_type)
    elif name == "qwen":
        return QwenEvaluator(embedding_key=embedding_key, evaluate_type=evaluate_type, query_type=query_type)
    elif name == "gemma":
        return GemmaEvaluator(embedding_key=embedding_key, evaluate_type=evaluate_type, query_type=query_type)
    else:
        raise ValueError(f"Unknown evaluator type: {name}")


class LlamaEvaluator(BaseEvaluator):
    def __init__(self, model_id="meta-llama/Llama-3.3-70B-Instruct", embedding_key=None, evaluate_type="normal", query_type="random"):
        self.model_id = model_id
        self.embedding_key = embedding_key
        self.evaluate_type = evaluate_type
        self.query_type = query_type

        print(f"[Evaluator] Loading Llama model: {model_id}")
        print(f"[Evaluator] Evaluation type: {evaluate_type}")

        if embedding_key:
            print(f"[Evaluator] Using embedding_key: {embedding_key}")

        if evaluate_type == "question":
            valid_query_types = ["problem", "challenge", "solution", "results", "combined", "random"]
            if query_type not in valid_query_types:
                raise ValueError(f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_query_types)}")
            print(f"[Evaluator] Using query_type: {query_type}")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def evaluate_hierarchy(
        self,
        hierarchy_path: str,
        abstracts_folder: str,
        test_count: int = 5,
        output_file: str = "evaluation_results.csv",
        embedding_key: str = None,
        query_type: str = None,
        run_number: int = 1
    ):
        if embedding_key is not None:
            self.embedding_key = embedding_key
            print(f"[Evaluator] Using embedding_key: {self.embedding_key}")

        hierarchy_data = self._load_hierarchy(hierarchy_path)
        cluster_dict, paper_to_path, paper_title_dict = self._build_cluster_dict_and_paths(hierarchy_data)
        top_level_clusters = hierarchy_data["clusters"]

        level_correct = {}
        level_total = {}
        level0_cluster_correct = {}
        level0_cluster_total = {}
        level0_errors = []

        max_depth = 0
        for pid, path_ in paper_to_path.items():
            if len(path_)>max_depth:
                max_depth = len(path_)

        for lv in range(max_depth):
            level_correct[lv] = 0
            level_total[lv] = 0

        abs_path = Path(abstracts_folder)
        all_files = [f for f in abs_path.iterdir() if f.is_file() and (f.suffix==".txt" or f.suffix==".json")]
        if not all_files:
            print(f"No .txt or .json files found in folder: {abstracts_folder}")
            return

        random.shuffle(all_files)
        test_files = all_files[:test_count]
        tested_count = 0

        for test_file in tqdm(test_files, desc="Evaluating papers"):
            test_file_path = str(test_file)

            if self.evaluate_type == "normal":
                paper_title, paper_content = self._extract_paper_content(test_file_path)
            else:
                paper_title, paper_structured_data = self._extract_paper_structured_data(test_file_path)

            if paper_title not in paper_title_dict:
                print(f"[WARNING] Could not find paper_id for title: {paper_title}")
                continue

            paper_id = paper_title_dict[paper_title]
            if paper_id not in paper_to_path:
                print(f"[WARNING] paper_id={paper_id} not found in path map")
                continue

            tested_count += 1
            true_path = paper_to_path[paper_id]

            if self.evaluate_type == "normal":
                print(f"\nEvaluating paper: {paper_title}")
            else:
                print(f"\nEvaluating search for paper: {paper_title}")

            print(f"True path: {true_path}")

            current_candidates = top_level_clusters

            for lv, cid_true in enumerate(true_path):
                level_total[lv]+=1
                if lv==0:
                    level0_cluster_total[cid_true] = level0_cluster_total.get(cid_true,0)+1

                real_cluster = None
                for c_ in current_candidates:
                    if c_.get("cluster_id")==cid_true:
                        real_cluster=c_
                        break
                if not real_cluster:
                    print(f"[ERROR] Could not find cluster {cid_true} in current candidates")
                    break

                if self.evaluate_type == "normal":
                    chosen_cid = self._ask_model_to_pick_cluster(
                        paper_title,
                        paper_content,
                        current_candidates
                    )
                else:
                    search_query = self._generate_search_query(paper_structured_data)
                    chosen_cid = self._ask_model_to_find_cluster(
                        search_query,
                        current_candidates
                    )
                    print(f"Query: {search_query}")

                print(f"Level {lv}: true={cid_true}, chosen={chosen_cid}")

                if lv==0 and chosen_cid!=cid_true:
                    real_title = real_cluster.get("title","Unknown")
                    chosen_title="Unknown"
                    for ccc in current_candidates:
                        if ccc.get("cluster_id")==chosen_cid:
                            chosen_title=ccc.get("title","Unknown")
                            break

                    ecase = {
                        "paper_title": paper_title,
                        "true_cluster": {
                            "id": cid_true,
                            "title": real_title
                        },
                        "predicted_cluster": {
                            "id": chosen_cid,
                            "title": chosen_title
                        }
                    }

                    if self.evaluate_type == "normal":
                        ecase["paper_content"] = paper_content[:200]+"..."
                    else:
                        ecase["search_query"] = search_query

                    level0_errors.append(ecase)

                if chosen_cid==-1:
                    break

                if chosen_cid==cid_true:
                    level_correct[lv]+=1
                    if lv==0:
                        level0_cluster_correct[cid_true] = level0_cluster_correct.get(cid_true,0)+1
                    if "children" not in real_cluster:
                        break
                    current_candidates=real_cluster["children"]
                else:
                    break

        if tested_count==0:
            print("\nNo valid test samples were tested.")
            return

        print("\n=== Evaluation Results ===")
        print(f"Tested papers: {tested_count}")
        for lv_ in range(max_depth):
            tot_ = level_total[lv_]
            if tot_>0:
                cor_ = level_correct[lv_]
                print(f"Level-{lv_} Accuracy: {cor_/tot_:.2%} ({cor_}/{tot_})")

        print("\nLevel 0 Cluster Accuracies:")
        for c_ in sorted(level0_cluster_total.keys()):
            t_ = level0_cluster_total[c_]
            c__=level0_cluster_correct.get(c_,0)
            if t_>0:
                print(f"Cluster {c_}: {c__}/{t_} = {c__/t_:.2%}")

        print("\n=== Level 0 Error Cases ===")
        print(f"Total error cases: {len(level0_errors)}")
        for i,err in enumerate(level0_errors,1):
            print(f"\nError Case {i}:")
            print(f"Paper: {err['paper_title']}")

            if self.evaluate_type == "normal":
                print(f"Content: {err['paper_content']}")
            else:
                print(f"Query: {err['search_query']}")

            print(f"True Cluster: {err['true_cluster']['id']} - {err['true_cluster']['title']}")
            print(f"Predicted Cluster: {err['predicted_cluster']['id']} - {err['predicted_cluster']['title']}")
            print("-"*80)

        h_name = Path(hierarchy_path).name
        self._write_results_csv(h_name, tested_count, level_total, level_correct, output_file, run_number)

        return {
            "accuracy": {lvl: level_correct[lvl]/level_total[lvl] if level_total[lvl] > 0 else 0
                        for lvl in range(max_depth)},
            "tested_count": tested_count,
            "level_correct": level_correct,
            "level_total": level_total
        }

    def _extract_paper_content(self, file_path):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as ff:
                content = ff.read()
            t_m = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|\\Z)", content, flags=re.MULTILINE|re.DOTALL)
            a_m = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            p_title = t_m.group(1).strip() if t_m else "Unknown Title"
            p_content = a_m.group(1).strip() if a_m else "Unknown Abstract"

        elif file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    if data:
                        data = data[0]
                    else:
                        return "Unknown Title", "Empty JSON file"

                p_title = data.get("title", "Unknown Title").strip()

                if self.embedding_key == "problem" and "problem" in data:
                    problem_data = data["problem"]
                    if isinstance(problem_data, dict):
                        p_content = json.dumps(problem_data, ensure_ascii=False)
                    else:
                        p_content = str(problem_data)

                elif self.embedding_key == "solution" and "solution" in data:
                    solution_data = data["solution"]
                    if isinstance(solution_data, dict):
                        p_content = json.dumps(solution_data, ensure_ascii=False)
                    else:
                        p_content = str(solution_data)

                elif self.embedding_key == "results" and "results" in data:
                    results_data = data["results"]
                    if isinstance(results_data, dict):
                        p_content = json.dumps(results_data, ensure_ascii=False)
                    else:
                        p_content = str(results_data)

                else:
                    p_content = data.get("abstract", "No relevant content found").strip()

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")
                return "Error parsing JSON", f"Error: {str(e)}"
        else:
            return "Unknown File Type", f"Unsupported file type: {file_path}"

        return p_title, p_content

    def _extract_paper_structured_data(self, file_path):
        title = "Unknown Title"
        structured_data = {
            "problem": {"overarching problem domain": "unknown", "challenges/difficulties": "unknown", "research question/goal": "unknown"},
            "solution": {"overarching solution domain": "unknown", "solution approach": "unknown", "novelty of the solution": "unknown"},
            "results": {"findings/results": "unknown", "potential impact of the results": "unknown"}
        }

        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    data = data[0]

                title = data.get("title", "Unknown Title").strip()

                if "problem" in data:
                    structured_data["problem"] = data["problem"]
                if "solution" in data:
                    structured_data["solution"] = data["solution"]
                if "results" in data:
                    structured_data["results"] = data["results"]

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            title_match = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|$)", content, flags=re.MULTILINE|re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()

            abstract_match = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                structured_data["problem"]["overarching problem domain"] = abstract[:100] + "..."
                structured_data["problem"]["research question/goal"] = abstract[100:200] + "..." if len(abstract) > 100 else abstract

        else:
            print(f"Unsupported file type: {file_path}")

        return title, structured_data

    def _generate_search_query(self, paper_data):
        domain = paper_data.get("problem", {}).get("overarching problem domain", "unknown")
        challenges = paper_data.get("problem", {}).get("challenges/difficulties", "unknown")
        research_goal = paper_data.get("problem", {}).get("research question/goal", "unknown")
        solution_domain = paper_data.get("solution", {}).get("overarching solution domain", "unknown")
        solution_approach = paper_data.get("solution", {}).get("solution approach", "unknown")
        novelty = paper_data.get("solution", {}).get("novelty of the solution", "unknown")
        findings = paper_data.get("results", {}).get("findings/results", "unknown")
        impact = paper_data.get("results", {}).get("potential impact of the results", "unknown")

        query_type = self.query_type
        if query_type == "random" and self.embedding_key:
            if self.embedding_key == "problem":
                query_type = random.choice(["problem", "challenge"])
            elif self.embedding_key == "solution":
                query_type = "solution"
            elif self.embedding_key == "results":
                query_type = "result"
            else:
                query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])
        elif query_type == "random":
            query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])

        if query_type == "problem":
            templates = [
                f"I'm looking for papers about {domain} that aim to {research_goal}.",
                f"Can you help me find research on {domain}? I'm particularly interested in {research_goal}.",
                f"I need to find papers focusing on {domain}, specifically those trying to {research_goal}.",
                f"Where can I find studies about {research_goal} in the field of {domain}?",
                f"I'm researching {domain} and want to learn more about approaches to {research_goal}."
            ]

        elif query_type == "challenge":
            templates = [
                f"I need papers addressing the challenge of {challenges} in {domain}.",
                f"What research tackles {challenges}? This is a key issue in {domain}.",
                f"I'm struggling with {challenges} in my research on {domain}. Any relevant papers?",
                f"Are there papers discussing solutions to {challenges} when working with {domain}?",
                f"My team is facing difficulties with {challenges}. What research exists on this problem?"
            ]

        elif query_type == "solution":
            templates = [
                f"I'm interested in papers that use {solution_approach} for problems in {domain}.",
                f"Can you point me to research utilizing {solution_approach}? The novelty of {novelty} sounds promising.",
                f"I want to learn more about {solution_domain} approaches, particularly {solution_approach}.",
                f"Are there papers describing {solution_approach} with {novelty}?",
                f"I'm looking for innovative approaches like {solution_approach} that offer {novelty}."
            ]

        elif query_type == "result":
            templates = [
                f"I need papers that have demonstrated {findings} in the field of {domain}.",
                f"Looking for research with potential impact on {impact}, particularly studies showing {findings}.",
                f"Can you recommend papers that achieved {findings} with implications for {impact}?",
                f"I'm interested in studies that reported {findings} and their applications.",
                f"What research has successfully shown {findings} that could influence {impact}?"
            ]

        elif query_type == "combined":
            templates = [
                f"I'm researching {domain} and need papers that use {solution_approach} to achieve {findings}.",
                f"Can you help me find studies addressing {challenges} through {solution_approach}?",
                f"I need papers on {domain} that demonstrate {findings} using novel approaches like {solution_approach}.",
                f"Looking for research that tackles {research_goal} and shows promising results in {findings}.",
                f"I'm interested in how {solution_approach} can be applied to {domain} problems, especially those showing {findings}."
            ]

        return random.choice(templates)

    def _ask_model_to_pick_cluster(self, paper_title, paper_content, candidate_clusters):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        content_type = "Abstract"
        if self.embedding_key == "problem":
            content_type = "Problem Statement"
        elif self.embedding_key == "solution":
            content_type = "Solution Approach"
        elif self.embedding_key == "results":
            content_type = "Research Results"

        user_content = f"""You are a scientist expert in taxonomy. Please read the following paper title and abstract.
Paper Title: {paper_title}
Paper {content_type}: {paper_content}

We have {len(valid_clusters)} candidate clusters:

{chr(10).join(cluster_descriptions)}

Please pick exactly one cluster ID that best matches the paper's content."""
        template = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"+
                    "{}\n\n Only return the final cluster ID, don't output anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe most appropriate cluster is Cluster ")
        full_prompt = template.format(user_content)

        print(full_prompt, flush=True)
        outs = self.pipeline(full_prompt, max_new_tokens=256)
        ftxt = outs[0]["generated_text"]
        response = ftxt[len(full_prompt):]


        m_ = re.search(r"\b(\d+)\b", response)
        return int(m_.group(1)) if m_ else -1

    def _ask_model_to_find_cluster(self, search_query, candidate_clusters):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        user_content = f"""You are a research librarian helping users find relevant papers in a taxonomy of research clusters.
A user has submitted the following search query:

"{search_query}"

Based on this query, which of the following clusters would most likely contain the papers they're looking for?

{chr(10).join(cluster_descriptions)}

Please identify the single most relevant cluster by its ID number."""
        template = ("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"+
                    "{}\n\n Only return the cluster ID number, don't output anything else.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nThe most relevant cluster is Cluster ")
        full_prompt = template.format(user_content)

        outs = self.pipeline(full_prompt, max_new_tokens=256)
        ftxt = outs[0]["generated_text"]
        response = ftxt[len(full_prompt):]

        m_ = re.search(r"\b(\d+)\b", response)
        return int(m_.group(1)) if m_ else -1

    def _load_hierarchy(self, path_):
        with open(path_, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_cluster_dict_and_paths(self, data_):
        c_dict = {}
        paper_to_path = {}
        paper_title_dict = {}

        def traverse_tree(node, curpath):
            if "cluster_id" in node:
                cid = node["cluster_id"]
                c_dict[cid] = node
                newp = curpath + [cid]
                if "children" in node:
                    for child in node["children"]:
                        if "cluster_id" in child:
                            traverse_tree(child, newp)
                        elif "paper_id" in child:
                            pid = child["paper_id"]
                            paper_to_path[pid] = newp
                            p_title = child.get("title", "Unknown").strip()
                            paper_title_dict[p_title] = pid
            else:
                print("Node missing cluster_id:", node)

        for top_c in data_["clusters"]:
            traverse_tree(top_c, [])

        return c_dict, paper_to_path, paper_title_dict

    def _write_results_csv(self, hname, tested_count, level_total, level_correct, out_csv, run_number=1):
        max_lv = 0 if not level_total else max(level_total.keys())
        results = {
            "hierarchy": hname,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluate_type": self.evaluate_type,
            "embedding_key": self.embedding_key or "none",
            "query_type": self.query_type if hasattr(self, 'query_type') and self.evaluate_type == "question" else "none",
            "run_number": run_number,
            "total_tested": tested_count
        }

        aggregate_score = sum(level_correct.values())
        max_possible_score = sum(level_total.values())
        aggregate_accuracy = aggregate_score / max_possible_score if max_possible_score > 0 else 0

        for lv in sorted(level_total.keys()):
            t_ = level_total[lv]
            if t_ > 0:
                c_ = level_correct[lv]
                results[f"level_{lv}_accuracy"] = f"{c_/t_:.4f}"
                results[f"level_{lv}_correct"] = c_
                results[f"level_{lv}_total"] = t_
            else:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = 0
                results[f"level_{lv}_total"] = 0

        results["aggregate_score"] = aggregate_score
        results["max_possible_score"] = max_possible_score
        results["aggregate_accuracy"] = f"{aggregate_accuracy:.4f}"

        file_exists = os.path.exists(out_csv)

        existing_max_lv = 0
        if file_exists:
            try:
                with open(out_csv, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            if field.startswith("level_") and field.endswith("_accuracy"):
                                level_num = int(field.split("_")[1])
                                existing_max_lv = max(existing_max_lv, level_num)
            except Exception:
                existing_max_lv = 0

        final_max_lv = max(max_lv, existing_max_lv)

        complete_fieldnames = ["hierarchy", "datetime", "evaluate_type", "embedding_key", "query_type", "run_number", "total_tested"]
        for lvv in range(final_max_lv + 1):
            complete_fieldnames.extend([
                f"level_{lvv}_accuracy",
                f"level_{lvv}_correct",
                f"level_{lvv}_total"
            ])
        complete_fieldnames.extend(["aggregate_score", "max_possible_score", "aggregate_accuracy"])

        for lv in range(final_max_lv + 1):
            if lv not in level_total:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = "none"
                results[f"level_{lv}_total"] = "none"

        if file_exists:
            with open(out_csv, 'r', newline='') as fh:
                reader = csv.DictReader(fh)
                existing_fieldnames = list(reader.fieldnames) if reader.fieldnames else []

                if set(complete_fieldnames) - set(existing_fieldnames):
                    print(f"Adding new columns for level(s) up to {final_max_lv}")

                    temp_csv = out_csv + ".temp"
                    with open(out_csv, 'r', newline='') as fin, open(temp_csv, 'w', newline='') as fout:
                        reader = csv.DictReader(fin)
                        writer = csv.DictWriter(fout, fieldnames=complete_fieldnames)
                        writer.writeheader()

                        for row in reader:
                            for field in complete_fieldnames:
                                if field not in row:
                                    row[field] = "none"
                            writer.writerow(row)

                    os.replace(temp_csv, out_csv)
                    fieldnames = complete_fieldnames
                else:
                    fieldnames = existing_fieldnames
        else:
            fieldnames = complete_fieldnames

        with open(out_csv, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)


class GemmaEvaluator(BaseEvaluator):
    def __init__(self, model_id="google/gemma-3-27b-it", embedding_key=None, evaluate_type="normal", query_type="random"):
        self.model_id = model_id
        self.embedding_key = embedding_key
        self.evaluate_type = evaluate_type
        self.query_type = query_type

        print(f"[Evaluator] Loading Gemma model: {model_id}")
        print(f"[Evaluator] Evaluation type: {evaluate_type}")

        if embedding_key:
            print(f"[Evaluator] Using embedding_key: {embedding_key}")

        if evaluate_type == "question":
            valid_query_types = ["problem", "challenge", "solution", "results", "combined", "random"]
            if query_type not in valid_query_types:
                raise ValueError(f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_query_types)}")
            print(f"[Evaluator] Using query_type: {query_type}")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            pad_to_multiple_of=8
        )

    def evaluate_hierarchy(
        self,
        hierarchy_path: str,
        abstracts_folder: str,
        test_count: int = 5,
        output_file: str = "evaluation_results.csv",
        embedding_key: str = None,
        query_type: str = None,
        run_number: int = 1
    ):
        if embedding_key is not None:
            self.embedding_key = embedding_key
            print(f"[Evaluator] Using embedding_key: {self.embedding_key}")

        hierarchy_data = self._load_hierarchy(hierarchy_path)
        cluster_dict, paper_to_path, paper_title_dict = self._build_cluster_dict_and_paths(hierarchy_data)
        top_level_clusters = hierarchy_data["clusters"]

        level_correct = {}
        level_total = {}
        level0_cluster_correct = {}
        level0_cluster_total = {}
        level0_errors = []

        max_depth = 0
        for pid, path_ in paper_to_path.items():
            if len(path_)>max_depth:
                max_depth = len(path_)

        for lv in range(max_depth):
            level_correct[lv] = 0
            level_total[lv] = 0

        abs_path = Path(abstracts_folder)
        all_files = [f for f in abs_path.iterdir() if f.is_file() and (f.suffix==".txt" or f.suffix==".json")]
        if not all_files:
            print(f"No .txt or .json files found in folder: {abstracts_folder}")
            return

        random.shuffle(all_files)
        test_files = all_files[:test_count]
        tested_count = 0

        for test_file in tqdm(test_files, desc="Evaluating papers"):
            test_file_path = str(test_file)

            if self.evaluate_type == "normal":
                paper_title, paper_content = self._extract_paper_content(test_file_path)
            else:
                paper_title, paper_structured_data = self._extract_paper_structured_data(test_file_path)

            if paper_title not in paper_title_dict:
                print(f"[WARNING] Could not find paper_id for title: {paper_title}")
                continue

            paper_id = paper_title_dict[paper_title]
            if paper_id not in paper_to_path:
                print(f"[WARNING] paper_id={paper_id} not found in path map")
                continue

            tested_count += 1
            true_path = paper_to_path[paper_id]

            if self.evaluate_type == "normal":
                print(f"\nEvaluating paper: {paper_title}")
            else:
                print(f"\nEvaluating search for paper: {paper_title}")

            print(f"True path: {true_path}")

            current_candidates = top_level_clusters

            for lv, cid_true in enumerate(true_path):
                level_total[lv]+=1
                if lv==0:
                    level0_cluster_total[cid_true] = level0_cluster_total.get(cid_true,0)+1

                real_cluster = None
                for c_ in current_candidates:
                    if c_.get("cluster_id")==cid_true:
                        real_cluster=c_
                        break
                if not real_cluster:
                    print(f"[ERROR] Could not find cluster {cid_true} in current candidates")
                    break

                if self.evaluate_type == "normal":
                    chosen_cid = self._ask_model_to_pick_cluster(
                        paper_title,
                        paper_content,
                        current_candidates
                    )
                else:
                    search_query = self._generate_search_query(paper_structured_data)
                    chosen_cid = self._ask_model_to_find_cluster(
                        search_query,
                        current_candidates
                    )
                    print(f"Query: {search_query}")

                print(f"Level {lv}: true={cid_true}, chosen={chosen_cid}")

                if lv==0 and chosen_cid!=cid_true:
                    real_title = real_cluster.get("title","Unknown")
                    chosen_title="Unknown"
                    for ccc in current_candidates:
                        if ccc.get("cluster_id")==chosen_cid:
                            chosen_title=ccc.get("title","Unknown")
                            break

                    ecase = {
                        "paper_title": paper_title,
                        "true_cluster": {
                            "id": cid_true,
                            "title": real_title
                        },
                        "predicted_cluster": {
                            "id": chosen_cid,
                            "title": chosen_title
                        }
                    }

                    if self.evaluate_type == "normal":
                        ecase["paper_content"] = paper_content[:200]+"..."
                    else:
                        ecase["search_query"] = search_query

                    level0_errors.append(ecase)

                if chosen_cid==-1:
                    break

                if chosen_cid==cid_true:
                    level_correct[lv]+=1
                    if lv==0:
                        level0_cluster_correct[cid_true] = level0_cluster_correct.get(cid_true,0)+1
                    if "children" not in real_cluster:
                        break
                    current_candidates=real_cluster["children"]
                else:
                    break

        if tested_count==0:
            print("\nNo valid test samples were tested.")
            return

        print("\n=== Evaluation Results ===")
        print(f"Tested papers: {tested_count}")
        for lv_ in range(max_depth):
            tot_ = level_total[lv_]
            if tot_>0:
                cor_ = level_correct[lv_]
                print(f"Level-{lv_} Accuracy: {cor_/tot_:.2%} ({cor_}/{tot_})")

        print("\nLevel 0 Cluster Accuracies:")
        for c_ in sorted(level0_cluster_total.keys()):
            t_ = level0_cluster_total[c_]
            c__=level0_cluster_correct.get(c_,0)
            if t_>0:
                print(f"Cluster {c_}: {c__}/{t_} = {c__/t_:.2%}")

        print("\n=== Level 0 Error Cases ===")
        print(f"Total error cases: {len(level0_errors)}")
        for i,err in enumerate(level0_errors,1):
            print(f"\nError Case {i}:")
            print(f"Paper: {err['paper_title']}")

            if self.evaluate_type == "normal":
                print(f"Content: {err['paper_content']}")
            else:
                print(f"Query: {err['search_query']}")

            print(f"True Cluster: {err['true_cluster']['id']} - {err['true_cluster']['title']}")
            print(f"Predicted Cluster: {err['predicted_cluster']['id']} - {err['predicted_cluster']['title']}")
            print("-"*80)

        h_name = Path(hierarchy_path).name
        self._write_results_csv(h_name, tested_count, level_total, level_correct, output_file, run_number)

        return {
            "accuracy": {lvl: level_correct[lvl]/level_total[lvl] if level_total[lvl] > 0 else 0
                        for lvl in range(max_depth)},
            "tested_count": tested_count,
            "level_correct": level_correct,
            "level_total": level_total
        }

    def _extract_paper_content(self, file_path):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as ff:
                content = ff.read()
            t_m = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|\\Z)", content, flags=re.MULTILINE|re.DOTALL)
            a_m = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            p_title = t_m.group(1).strip() if t_m else "Unknown Title"
            p_content = a_m.group(1).strip() if a_m else "Unknown Abstract"

        elif file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    if data:
                        data = data[0]
                    else:
                        return "Unknown Title", "Empty JSON file"

                p_title = data.get("title", "Unknown Title").strip()

                if self.embedding_key == "problem" and "problem" in data:
                    problem_data = data["problem"]
                    if isinstance(problem_data, dict):
                        p_content = json.dumps(problem_data, ensure_ascii=False)
                    else:
                        p_content = str(problem_data)

                elif self.embedding_key == "solution" and "solution" in data:
                    solution_data = data["solution"]
                    if isinstance(solution_data, dict):
                        p_content = json.dumps(solution_data, ensure_ascii=False)
                    else:
                        p_content = str(solution_data)

                elif self.embedding_key == "results" and "results" in data:
                    results_data = data["results"]
                    if isinstance(results_data, dict):
                        p_content = json.dumps(results_data, ensure_ascii=False)
                    else:
                        p_content = str(results_data)

                else:
                    p_content = data.get("abstract", "No relevant content found").strip()

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")
                return "Error parsing JSON", f"Error: {str(e)}"
        else:
            return "Unknown File Type", f"Unsupported file type: {file_path}"

        return p_title, p_content

    def _extract_paper_structured_data(self, file_path):
        title = "Unknown Title"
        structured_data = {
            "problem": {"overarching problem domain": "unknown", "challenges/difficulties": "unknown", "research question/goal": "unknown"},
            "solution": {"overarching solution domain": "unknown", "solution approach": "unknown", "novelty of the solution": "unknown"},
            "results": {"findings/results": "unknown", "potential impact of the results": "unknown"}
        }

        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    data = data[0]

                title = data.get("title", "Unknown Title").strip()

                if "problem" in data:
                    structured_data["problem"] = data["problem"]
                if "solution" in data:
                    structured_data["solution"] = data["solution"]
                if "results" in data:
                    structured_data["results"] = data["results"]

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            title_match = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|$)", content, flags=re.MULTILINE|re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()

            abstract_match = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                structured_data["problem"]["overarching problem domain"] = abstract[:100] + "..."
                structured_data["problem"]["research question/goal"] = abstract[100:200] + "..." if len(abstract) > 100 else abstract

        else:
            print(f"Unsupported file type: {file_path}")

        return title, structured_data

    def _generate_search_query(self, paper_data):
        domain = paper_data.get("problem", {}).get("overarching problem domain", "unknown")
        challenges = paper_data.get("problem", {}).get("challenges/difficulties", "unknown")
        research_goal = paper_data.get("problem", {}).get("research question/goal", "unknown")
        solution_domain = paper_data.get("solution", {}).get("overarching solution domain", "unknown")
        solution_approach = paper_data.get("solution", {}).get("solution approach", "unknown")
        novelty = paper_data.get("solution", {}).get("novelty of the solution", "unknown")
        findings = paper_data.get("results", {}).get("findings/results", "unknown")
        impact = paper_data.get("results", {}).get("potential impact of the results", "unknown")

        query_type = self.query_type
        if query_type == "random" and self.embedding_key:
            if self.embedding_key == "problem":
                query_type = random.choice(["problem", "challenge"])
            elif self.embedding_key == "solution":
                query_type = "solution"
            elif self.embedding_key == "results":
                query_type = "result"
            else:
                query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])
        elif query_type == "random":
            query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])

        if query_type == "problem":
            templates = [
                f"I'm looking for papers about {domain} that aim to {research_goal}.",
                f"Can you help me find research on {domain}? I'm particularly interested in {research_goal}.",
                f"I need to find papers focusing on {domain}, specifically those trying to {research_goal}.",
                f"Where can I find studies about {research_goal} in the field of {domain}?",
                f"I'm researching {domain} and want to learn more about approaches to {research_goal}."
            ]

        elif query_type == "challenge":
            templates = [
                f"I need papers addressing the challenge of {challenges} in {domain}.",
                f"What research tackles {challenges}? This is a key issue in {domain}.",
                f"I'm struggling with {challenges} in my research on {domain}. Any relevant papers?",
                f"Are there papers discussing solutions to {challenges} when working with {domain}?",
                f"My team is facing difficulties with {challenges}. What research exists on this problem?"
            ]

        elif query_type == "solution":
            templates = [
                f"I'm interested in papers that use {solution_approach} for problems in {domain}.",
                f"Can you point me to research utilizing {solution_approach}? The novelty of {novelty} sounds promising.",
                f"I want to learn more about {solution_domain} approaches, particularly {solution_approach}.",
                f"Are there papers describing {solution_approach} with {novelty}?",
                f"I'm looking for innovative approaches like {solution_approach} that offer {novelty}."
            ]

        elif query_type == "result":
            templates = [
                f"I need papers that have demonstrated {findings} in the field of {domain}.",
                f"Looking for research with potential impact on {impact}, particularly studies showing {findings}.",
                f"Can you recommend papers that achieved {findings} with implications for {impact}?",
                f"I'm interested in studies that reported {findings} and their applications.",
                f"What research has successfully shown {findings} that could influence {impact}?"
            ]

        elif query_type == "combined":
            templates = [
                f"I'm researching {domain} and need papers that use {solution_approach} to achieve {findings}.",
                f"Can you help me find studies addressing {challenges} through {solution_approach}?",
                f"I need papers on {domain} that demonstrate {findings} using novel approaches like {solution_approach}.",
                f"Looking for research that tackles {research_goal} and shows promising results in {findings}.",
                f"I'm interested in how {solution_approach} can be applied to {domain} problems, especially those showing {findings}."
            ]

        return random.choice(templates)

    def _ask_model_to_pick_cluster(self, paper_title, paper_content, candidate_clusters):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        content_type = "Abstract"
        if self.embedding_key == "problem":
            content_type = "Problem Statement"
        elif self.embedding_key == "solution":
            content_type = "Solution Approach"
        elif self.embedding_key == "results":
            content_type = "Research Results"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a scientist expert in taxonomy. Please read the following paper and clusters, pick one cluster ID that best matches the paper's content.
Paper Title: {paper_title}
Paper {content_type}: {paper_content}

We have {len(valid_clusters)} candidate clusters:

{chr(10).join(cluster_descriptions)}

Please pick exactly one cluster ID that best matches the paper's content. Only return the final cluster ID, don't output anything else."""
                    }
                ]
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        print(messages, flush=True)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)

        m_ = re.search(r"\b(\d+)\b", response)
        return int(m_.group(1)) if m_ else -1

    def _ask_model_to_find_cluster(self, search_query, candidate_clusters):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are a research librarian helping users find relevant papers in a taxonomy of research clusters.
A user has submitted the following search query:

"{search_query}"

Based on this query, which of the following clusters would most likely contain the papers they're looking for?

{chr(10).join(cluster_descriptions)}

Please identify the single most relevant cluster by its ID number. Only return the cluster ID number, don't output anything else."""
                    }
                ]
            }
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        outputs = self.model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=False
        )

        response = self.tokenizer.decode(outputs[0][inputs.size(1):], skip_special_tokens=True)

        m_ = re.search(r"\b(\d+)\b", response)
        return int(m_.group(1)) if m_ else -1

    def _load_hierarchy(self, path_):
        with open(path_, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_cluster_dict_and_paths(self, data_):
        c_dict = {}
        paper_to_path = {}
        paper_title_dict = {}

        def traverse_tree(node, curpath):
            if "cluster_id" in node:
                cid = node["cluster_id"]
                c_dict[cid] = node
                newp = curpath + [cid]
                if "children" in node:
                    for child in node["children"]:
                        if "cluster_id" in child:
                            traverse_tree(child, newp)
                        elif "paper_id" in child:
                            pid = child["paper_id"]
                            paper_to_path[pid] = newp
                            p_title = child.get("title", "Unknown").strip()
                            paper_title_dict[p_title] = pid
            else:
                print("Node missing cluster_id:", node)

        for top_c in data_["clusters"]:
            traverse_tree(top_c, [])

        return c_dict, paper_to_path, paper_title_dict

    def _write_results_csv(self, hname, tested_count, level_total, level_correct, out_csv, run_number=1):
        max_lv = 0 if not level_total else max(level_total.keys())
        results = {
            "hierarchy": hname,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluate_type": self.evaluate_type,
            "embedding_key": self.embedding_key or "none",
            "query_type": self.query_type if hasattr(self, 'query_type') and self.evaluate_type == "question" else "none",
            "run_number": run_number,
            "total_tested": tested_count
        }

        aggregate_score = sum(level_correct.values())
        max_possible_score = sum(level_total.values())
        aggregate_accuracy = aggregate_score / max_possible_score if max_possible_score > 0 else 0

        for lv in sorted(level_total.keys()):
            t_ = level_total[lv]
            if t_ > 0:
                c_ = level_correct[lv]
                results[f"level_{lv}_accuracy"] = f"{c_/t_:.4f}"
                results[f"level_{lv}_correct"] = c_
                results[f"level_{lv}_total"] = t_
            else:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = 0
                results[f"level_{lv}_total"] = 0

        results["aggregate_score"] = aggregate_score
        results["max_possible_score"] = max_possible_score
        results["aggregate_accuracy"] = f"{aggregate_accuracy:.4f}"

        file_exists = os.path.exists(out_csv)

        existing_max_lv = 0
        if file_exists:
            try:
                with open(out_csv, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            if field.startswith("level_") and field.endswith("_accuracy"):
                                level_num = int(field.split("_")[1])
                                existing_max_lv = max(existing_max_lv, level_num)
            except Exception:
                existing_max_lv = 0

        final_max_lv = max(max_lv, existing_max_lv)

        complete_fieldnames = ["hierarchy", "datetime", "evaluate_type", "embedding_key", "query_type", "run_number", "total_tested"]
        for lvv in range(final_max_lv + 1):
            complete_fieldnames.extend([
                f"level_{lvv}_accuracy",
                f"level_{lvv}_correct",
                f"level_{lvv}_total"
            ])
        complete_fieldnames.extend(["aggregate_score", "max_possible_score", "aggregate_accuracy"])

        for lv in range(final_max_lv + 1):
            if lv not in level_total:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = "none"
                results[f"level_{lv}_total"] = "none"

        if file_exists:
            with open(out_csv, 'r', newline='') as fh:
                reader = csv.DictReader(fh)
                existing_fieldnames = list(reader.fieldnames) if reader.fieldnames else []

                if set(complete_fieldnames) - set(existing_fieldnames):
                    print(f"Adding new columns for level(s) up to {final_max_lv}")

                    temp_csv = out_csv + ".temp"
                    with open(out_csv, 'r', newline='') as fin, open(temp_csv, 'w', newline='') as fout:
                        reader = csv.DictReader(fin)
                        writer = csv.DictWriter(fout, fieldnames=complete_fieldnames)
                        writer.writeheader()

                        for row in reader:
                            for field in complete_fieldnames:
                                if field not in row:
                                    row[field] = "none"
                            writer.writerow(row)

                    os.replace(temp_csv, out_csv)
                    fieldnames = complete_fieldnames
                else:
                    fieldnames = existing_fieldnames
        else:
            fieldnames = complete_fieldnames

        with open(out_csv, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)


class QwenEvaluator(BaseEvaluator):
    def __init__(self, model_id="Qwen/Qwen2.5-32B-Instruct", embedding_key=None, evaluate_type="normal", query_type="random"):
        self.model_id = model_id
        self.embedding_key = embedding_key
        self.evaluate_type = evaluate_type
        self.query_type = query_type

        print(f"[Evaluator] Loading Qwen model: {model_id}")
        print(f"[Evaluator] Evaluation type: {evaluate_type}")

        if embedding_key:
            print(f"[Evaluator] Using embedding_key: {embedding_key}")

        if evaluate_type == "question":
            valid_query_types = ["problem", "challenge", "solution", "results", "combined", "random"]
            if query_type not in valid_query_types:
                raise ValueError(f"Invalid query_type '{query_type}'. Must be one of: {', '.join(valid_query_types)}")
            print(f"[Evaluator] Using query_type: {query_type}")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def _write_detailed_results_csv(self, detailed_results, max_depth, output_file):
        if not detailed_results:
            print("No detailed results to write.")
            return

        fieldnames = ["paper_title", "paper_summary"]
        for lv in range(max_depth):
            fieldnames.extend([
                f"level_{lv}_candidates",
                f"level_{lv}_correct_id",
                f"level_{lv}_correct_title",
                f"level_{lv}_correct_summary",
                f"level_{lv}_predicted_id",
                f"level_{lv}_predicted_title",
                f"level_{lv}_predicted_summary"
            ])

        for result in detailed_results:
            for lv in range(max_depth):
                candidates_key = f"level_{lv}_candidates"
                if candidates_key in result and result[candidates_key] != "NA":
                    candidates = result[candidates_key]
                    candidate_strs = []
                    for cand in candidates:
                        candidate_strs.append(f"ID:{cand['cluster_id']}-Title:{cand['title'][:50]}")
                    result[candidates_key] = "; ".join(candidate_strs)

                for key in [f"level_{lv}_correct_summary", f"level_{lv}_predicted_summary"]:
                    if key in result and result[key] != "NA" and isinstance(result[key], str):
                        result[key] = result[key][:100] + "..." if len(result[key]) > 100 else result[key]

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in detailed_results:
                    writer.writerow(result)
            print(f"Detailed results written to {output_file}")
        except Exception as e:
            print(f"Error writing detailed results to CSV: {e}")

    def _write_error_cases_jsonl(self, error_cases, output_file):
        if not error_cases:
            return
        try:
            with open(output_file, 'a', encoding='utf-8') as f:
                for ec in error_cases:
                    f.write(json.dumps(ec, ensure_ascii=False) + "\n")
            print(f"Error cases appended to {output_file}")
        except Exception as e:
            print(f"Error writing error cases JSONL: {e}")

    def evaluate_hierarchy(
        self,
        hierarchy_path: str,
        abstracts_folder: str,
        test_count: int = 5,
        output_file: str = "evaluation_results.csv",
        embedding_key: str = None,
        query_type: str = None,
        detailed_output_file: str = "evaluation_details.csv",
        error_output_file: str = "error_cases.jsonl",
        run_number: int = 1
    ):
        if embedding_key is not None:
            self.embedding_key = embedding_key
            print(f"[Evaluator] Using embedding_key: {self.embedding_key}")

        hierarchy_data = self._load_hierarchy(hierarchy_path)
        cluster_dict, paper_to_path, paper_title_dict = self._build_cluster_dict_and_paths(hierarchy_data)
        top_level_clusters = hierarchy_data["clusters"]

        level_correct = {}
        level_total = {}
        level0_cluster_correct = {}
        level0_cluster_total = {}
        level0_errors = []
        structured_errors = []

        detailed_results = []

        max_depth = 0
        for pid, path_ in paper_to_path.items():
            if len(path_)>max_depth:
                max_depth = len(path_)

        for lv in range(max_depth):
            level_correct[lv] = 0
            level_total[lv] = 0

        abs_path = Path(abstracts_folder)
        all_files = [f for f in abs_path.iterdir() if f.is_file() and (f.suffix==".txt" or f.suffix==".json")]
        if not all_files:
            print(f"No .txt or .json files found in folder: {abstracts_folder}")
            return

        random.shuffle(all_files)
        test_files = all_files[:test_count]
        tested_count = 0

        for test_file in tqdm(test_files, desc="Evaluating papers"):
            test_file_path = str(test_file)

            if self.evaluate_type == "normal":
                paper_title, paper_content = self._extract_paper_content(test_file_path)
            else:
                paper_title, paper_structured_data = self._extract_paper_structured_data(test_file_path)
                paper_content = self._generate_search_query(paper_structured_data)

            if paper_title not in paper_title_dict:
                print(f"[WARNING] Could not find paper_id for title: {paper_title}")
                continue

            paper_id = paper_title_dict[paper_title]
            if paper_id not in paper_to_path:
                print(f"[WARNING] paper_id={paper_id} not found in path map")
                continue

            tested_count += 1
            true_path = paper_to_path[paper_id]

            if self.evaluate_type == "normal":
                print(f"\nEvaluating paper: {paper_title}")
            else:
                print(f"\nEvaluating search for paper: {paper_title}")

            print(f"True path: {true_path}")

            sample_result = {
                "paper_title": paper_title,
                "paper_summary": paper_content
            }

            current_candidates = top_level_clusters
            reached_level = 0

            for lv, cid_true in enumerate(true_path):
                level_total[lv]+=1
                if lv==0:
                    level0_cluster_total[cid_true] = level0_cluster_total.get(cid_true,0)+1

                real_cluster = None
                for c_ in current_candidates:
                    if c_.get("cluster_id")==cid_true:
                        real_cluster=c_
                        break
                if not real_cluster:
                    print(f"[ERROR] Could not find cluster {cid_true} in current candidates")
                    break

                candidates_info = []
                for c in current_candidates:
                    if "cluster_id" in c:
                        candidates_info.append({
                            "cluster_id": c.get("cluster_id"),
                            "title": c.get("title", ""),
                            "summary": c.get("abstract", "")
                        })

                sample_result[f"level_{lv}_candidates"] = candidates_info
                sample_result[f"level_{lv}_correct_id"] = cid_true
                sample_result[f"level_{lv}_correct_title"] = real_cluster.get("title", "")
                sample_result[f"level_{lv}_correct_summary"] = real_cluster.get("abstract", "")

                if self.evaluate_type == "normal":
                    ret = self._ask_model_to_pick_cluster(
                        paper_title,
                        paper_content,
                        current_candidates,
                        path_so_far=f"level_{lv}",
                        return_raw=True
                    )
                    chosen_cid = ret.get("chosen_id", -1)
                    model_prompt = ret.get("prompt")
                    model_response = ret.get("response")
                else:
                    search_query = self._generate_search_query(paper_structured_data)
                    ret = self._ask_model_to_find_cluster(
                        search_query,
                        current_candidates,
                        path_so_far=f"level_{lv}",
                        return_raw=True
                    )
                    chosen_cid = ret.get("chosen_id", -1)
                    model_prompt = ret.get("prompt")
                    model_response = ret.get("response")
                    print(f"Query: {search_query}")

                print(f"Level {lv}: true={cid_true}, chosen={chosen_cid}")
                sample_result[f"level_{lv}_predicted_id"] = chosen_cid

                chosen_cluster = None
                for c in current_candidates:
                    if c.get("cluster_id") == chosen_cid:
                        chosen_cluster = c
                        break

                if chosen_cluster:
                    sample_result[f"level_{lv}_predicted_title"] = chosen_cluster.get("title", "")
                    sample_result[f"level_{lv}_predicted_summary"] = chosen_cluster.get("abstract", "")
                else:
                    sample_result[f"level_{lv}_predicted_title"] = "Unknown"
                    sample_result[f"level_{lv}_predicted_summary"] = ""

                if lv==0 and chosen_cid!=cid_true:
                    real_title = real_cluster.get("title","Unknown")
                    chosen_title="Unknown"
                    for ccc in current_candidates:
                        if ccc.get("cluster_id")==chosen_cid:
                            chosen_title=ccc.get("title","Unknown")
                            break

                    ecase = {
                        "paper_title": paper_title,
                        "true_cluster": {
                            "id": cid_true,
                            "title": real_title
                        },
                        "predicted_cluster": {
                            "id": chosen_cid,
                            "title": chosen_title
                        }
                    }

                    if self.evaluate_type == "normal":
                        ecase["paper_content"] = paper_content[:200]+"..."
                    else:
                        ecase["search_query"] = search_query

                    level0_errors.append(ecase)

                    chosen_cluster = None
                    for ccc in current_candidates:
                        if ccc.get("cluster_id") == chosen_cid:
                            chosen_cluster = ccc
                            break
                    structured_errors.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "hierarchy": Path(hierarchy_path).name,
                        "evaluator": "qwen",
                        "model_id": self.model_id,
                        "evaluate_type": self.evaluate_type,
                        "embedding_key": self.embedding_key or "none",
                        "query_type": self.query_type or "none",
                        "run_number": run_number,
                        "paper_id": paper_id,
                        "paper_title": paper_title,
                        "paper_summary": paper_content,
                        "true_path": true_path,
                        "error_level": lv,
                        "candidates": [
                            {
                                "id": c.get("cluster_id"),
                                "title": c.get("title", ""),
                                "summary": c.get("abstract", "")
                            } for c in current_candidates if "cluster_id" in c
                        ],
                        "true_cluster": {
                            "id": cid_true,
                            "title": real_cluster.get("title", ""),
                            "summary": real_cluster.get("abstract", "")
                        },
                        "predicted_cluster": {
                            "id": chosen_cid,
                            "title": chosen_cluster.get("title", "") if chosen_cluster else "",
                            "summary": chosen_cluster.get("abstract", "") if chosen_cluster else ""
                        },
                        "prompt": model_prompt,
                        "response": model_response
                    })

                if chosen_cid==-1:
                    reached_level = lv
                    break

                if chosen_cid==cid_true:
                    level_correct[lv]+=1
                    if lv==0:
                        level0_cluster_correct[cid_true] = level0_cluster_correct.get(cid_true,0)+1
                    if "children" not in real_cluster:
                        reached_level = lv + 1
                        break
                    current_candidates=real_cluster["children"]
                else:
                    reached_level = lv
                    break

            for lv in range(reached_level + 1, max_depth):
                sample_result[f"level_{lv}_candidates"] = "NA"
                sample_result[f"level_{lv}_correct_id"] = "NA"
                sample_result[f"level_{lv}_correct_title"] = "NA"
                sample_result[f"level_{lv}_correct_summary"] = "NA"
                sample_result[f"level_{lv}_predicted_id"] = "NA"
                sample_result[f"level_{lv}_predicted_title"] = "NA"
                sample_result[f"level_{lv}_predicted_summary"] = "NA"

            detailed_results.append(sample_result)

        if tested_count==0:
            print("\nNo valid test samples were tested.")
            return

        print("\n=== Evaluation Results ===")
        print(f"Tested papers: {tested_count}")
        for lv_ in range(max_depth):
            tot_ = level_total[lv_]
            if tot_>0:
                cor_ = level_correct[lv_]
                print(f"Level-{lv_} Accuracy: {cor_/tot_:.2%} ({cor_}/{tot_})")

        print("\nLevel 0 Cluster Accuracies:")
        for c_ in sorted(level0_cluster_total.keys()):
            t_ = level0_cluster_total[c_]
            c__=level0_cluster_correct.get(c_,0)
            if t_>0:
                print(f"Cluster {c_}: {c__}/{t_} = {c__/t_:.2%}")

        print("\n=== Level 0 Error Cases ===")
        print(f"Total error cases: {len(level0_errors)}")
        for i,err in enumerate(level0_errors,1):
            print(f"\nError Case {i}:")
            print(f"Paper: {err['paper_title']}")

            if self.evaluate_type == "normal":
                print(f"Content: {err['paper_content']}")
            else:
                print(f"Query: {err['search_query']}")

            print(f"True Cluster: {err['true_cluster']['id']} - {err['true_cluster']['title']}")
            print(f"Predicted Cluster: {err['predicted_cluster']['id']} - {err['predicted_cluster']['title']}")
            print("-"*80)

        h_name = Path(hierarchy_path).name
        self._write_results_csv(h_name, tested_count, level_total, level_correct, output_file, run_number)

        self._write_detailed_results_csv(detailed_results, max_depth, detailed_output_file)
        self._write_error_cases_jsonl(structured_errors, error_output_file)

        return {
            "accuracy": {lvl: level_correct[lvl]/level_total[lvl] if level_total[lvl] > 0 else 0
                        for lvl in range(max_depth)},
            "tested_count": tested_count,
            "level_correct": level_correct,
            "level_total": level_total
        }

    def _extract_paper_content(self, file_path):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as ff:
                content = ff.read()
            t_m = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|\\Z)", content, flags=re.MULTILINE|re.DOTALL)
            a_m = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            p_title = t_m.group(1).strip() if t_m else "Unknown Title"
            p_content = a_m.group(1).strip() if a_m else "Unknown Abstract"

        elif file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    if data:
                        data = data[0]
                    else:
                        return "Unknown Title", "Empty JSON file"

                p_title = data.get("title", "Unknown Title").strip()

                if self.embedding_key == "problem" and "problem" in data:
                    problem_data = data["problem"]
                    if isinstance(problem_data, dict):
                        p_content = json.dumps(problem_data, ensure_ascii=False)
                    else:
                        p_content = str(problem_data)

                elif self.embedding_key == "solution" and "solution" in data:
                    solution_data = data["solution"]
                    if isinstance(solution_data, dict):
                        p_content = json.dumps(solution_data, ensure_ascii=False)
                    else:
                        p_content = str(solution_data)

                elif self.embedding_key == "results" and "results" in data:
                    results_data = data["results"]
                    if isinstance(results_data, dict):
                        p_content = json.dumps(results_data, ensure_ascii=False)
                    else:
                        p_content = str(results_data)

                else:
                    p_content = data.get("abstract", "No relevant content found").strip()

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")
                return "Error parsing JSON", f"Error: {str(e)}"
        else:
            return "Unknown File Type", f"Unsupported file type: {file_path}"

        p_content = data.get("abstract", "No relevant content found").strip()
        return p_title, p_content

    def _extract_paper_structured_data(self, file_path):
        title = "Unknown Title"
        structured_data = {
            "problem": {"overarching problem domain": "unknown", "challenges/difficulties": "unknown", "research question/goal": "unknown"},
            "solution": {"overarching solution domain": "unknown", "solution approach": "unknown", "novelty of the solution": "unknown"},
            "results": {"findings/results": "unknown", "potential impact of the results": "unknown"}
        }

        if file_path.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list) and data:
                    data = data[0]

                title = data.get("title", "Unknown Title").strip()

                if "problem" in data:
                    structured_data["problem"] = data["problem"]
                if "solution" in data:
                    structured_data["solution"] = data["solution"]
                if "results" in data:
                    structured_data["results"] = data["results"]

            except Exception as e:
                print(f"Error parsing JSON file {file_path}: {e}")

        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            title_match = re.search(r"(?i)^Title:\s*(.*?)(?=^Abstract:|$)", content, flags=re.MULTILINE|re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()

            abstract_match = re.search(r"(?i)^Abstract:\s*(.*)", content, flags=re.MULTILINE|re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                structured_data["problem"]["overarching problem domain"] = abstract[:100] + "..."
                structured_data["problem"]["research question/goal"] = abstract[100:200] + "..." if len(abstract) > 100 else abstract

        else:
            print(f"Unsupported file type: {file_path}")

        return title, structured_data

    def _generate_search_query(self, paper_data):
        domain = paper_data.get("problem", {}).get("overarching problem domain", "unknown")
        challenges = paper_data.get("problem", {}).get("challenges/difficulties", "unknown")
        research_goal = paper_data.get("problem", {}).get("research question/goal", "unknown")
        solution_domain = paper_data.get("solution", {}).get("overarching solution domain", "unknown")
        solution_approach = paper_data.get("solution", {}).get("solution approach", "unknown")
        novelty = paper_data.get("solution", {}).get("novelty of the solution", "unknown")
        findings = paper_data.get("results", {}).get("findings/results", "unknown")
        impact = paper_data.get("results", {}).get("potential impact of the results", "unknown")

        query_type = self.query_type
        if query_type == "random" and self.embedding_key:
            if self.embedding_key == "problem":
                query_type = random.choice(["problem", "challenge"])
            elif self.embedding_key == "solution":
                query_type = "solution"
            elif self.embedding_key == "results":
                query_type = "result"
            else:
                query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])
        elif query_type == "random":
            query_type = random.choice(["problem", "challenge", "solution", "result", "combined"])

        if query_type == "problem":
            templates = [
                f"I'm looking for papers about {domain} that aim to {research_goal}.",
                f"Can you help me find research on {domain}? I'm particularly interested in {research_goal}.",
                f"I need to find papers focusing on {domain}, specifically those trying to {research_goal}.",
                f"Where can I find studies about {research_goal} in the field of {domain}?",
                f"I'm researching {domain} and want to learn more about approaches to {research_goal}."
            ]

        elif query_type == "challenge":
            templates = [
                f"I need papers addressing the challenge of {challenges} in {domain}.",
                f"What research tackles {challenges}? This is a key issue in {domain}.",
                f"I'm struggling with {challenges} in my research on {domain}. Any relevant papers?",
                f"Are there papers discussing solutions to {challenges} when working with {domain}?",
                f"My team is facing difficulties with {challenges}. What research exists on this problem?"
            ]

        elif query_type == "solution":
            templates = [
                f"I'm interested in papers that use {solution_approach} for problems in {domain}.",
                f"Can you point me to research utilizing {solution_approach}? The novelty of {novelty} sounds promising.",
                f"I want to learn more about {solution_domain} approaches, particularly {solution_approach}.",
                f"Are there papers describing {solution_approach} with {novelty}?",
                f"I'm looking for innovative approaches like {solution_approach} that offer {novelty}."
            ]

        elif query_type == "result":
            templates = [
                f"I need papers that have demonstrated {findings} in the field of {domain}.",
                f"Looking for research with potential impact on {impact}, particularly studies showing {findings}.",
                f"Can you recommend papers that achieved {findings} with implications for {impact}?",
                f"I'm interested in studies that reported {findings} and their applications.",
                f"What research has successfully shown {findings} that could influence {impact}?"
            ]

        elif query_type == "combined":
            templates = [
                f"I'm researching {domain} and need papers that use {solution_approach} to achieve {findings}.",
                f"Can you help me find studies addressing {challenges} through {solution_approach}?",
                f"I need papers on {domain} that demonstrate {findings} using novel approaches like {solution_approach}.",
                f"Looking for research that tackles {research_goal} and shows promising results in {findings}.",
                f"I'm interested in how {solution_approach} can be applied to {domain} problems, especially those showing {findings}."
            ]

        return random.choice(templates)

    def _ask_model_to_pick_cluster(self, paper_title, paper_content, candidate_clusters, path_so_far="Root", return_raw: bool=False):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        content_type = "Abstract"
        if self.embedding_key == "problem":
            content_type = "Problem Statement"
        elif self.embedding_key == "solution":
            content_type = "Solution Approach"
        elif self.embedding_key == "results":
            content_type = "Research Results"

        messages = [
            {"role": "system", "content": "You are a scientist expert in taxonomy. Please read the following paper title and abstract."},
            {"role": "user", "content": f"""Your task is to choose the next cluster (while considering the current path) in the taxonomy that has the best chance of containing this paper.
Paper Title: {paper_title}
Paper {content_type}: {paper_content}

Current Path: {path_so_far}

Choose from this cluster list (MUST pick one):

{chr(10).join(cluster_descriptions)}

Required Response Format:
Cluster ID: [EXACT ID from the list]"""}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        print(messages, flush=True)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response, flush=True)
        m_ = re.search(r"\b(\d+)\b", response)
        chosen = int(m_.group(1)) if m_ else -1
        if return_raw:
            return {"chosen_id": chosen, "prompt": text, "response": response}
        return chosen

    def _ask_model_to_find_cluster(self, search_query, candidate_clusters, path_so_far="Root", return_raw: bool=False):
        cluster_descriptions = []
        valid_clusters = []
        for c in candidate_clusters:
            if "cluster_id" not in c:
                print("Skipping node (no cluster_id)", c)
                continue
            c_id = c["cluster_id"]
            c_t = c.get("title", "")
            c_a = c.get("abstract", "")
            cluster_descriptions.append(f"- Cluster ID: {c_id}\n  Title: {c_t}\n  Summary: {c_a}")
            valid_clusters.append(c)
        if not valid_clusters:
            print("No valid sub-clusters found.")
            return -1

        messages = [
            {"role": "system", "content": "You are a research librarian helping users find relevant papers in a taxonomy of research clusters."},
            {"role": "user", "content": f"""A user has submitted the following search query:

"{search_query}"

Based on this query, which of the following clusters would most likely contain the papers they're looking for?

Current Path in Taxonomy: {path_so_far}

Available clusters:
{chr(10).join(cluster_descriptions)}

Please identify the single most relevant cluster by its ID number. Only return the cluster ID number, don't output anything else."""}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=256,
            do_sample=False
        )

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        m_ = re.search(r"\b(\d+)\b", response)
        chosen = int(m_.group(1)) if m_ else -1
        if return_raw:
            return {"chosen_id": chosen, "prompt": text, "response": response}
        return chosen

    def _load_hierarchy(self, path_):
        with open(path_, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _build_cluster_dict_and_paths(self, data_):
        c_dict = {}
        paper_to_path = {}
        paper_title_dict = {}

        def traverse_tree(node, curpath):
            if "cluster_id" in node:
                cid = node["cluster_id"]
                c_dict[cid] = node
                newp = curpath + [cid]
                if "children" in node:
                    for child in node["children"]:
                        if "cluster_id" in child:
                            traverse_tree(child, newp)
                        elif "paper_id" in child:
                            pid = child["paper_id"]
                            paper_to_path[pid] = newp
                            p_title = child.get("title", "Unknown").strip()
                            paper_title_dict[p_title] = pid
            else:
                print("Node missing cluster_id:", node)

        for top_c in data_["clusters"]:
            traverse_tree(top_c, [])

        return c_dict, paper_to_path, paper_title_dict

    def _write_results_csv(self, hname, tested_count, level_total, level_correct, out_csv, run_number=1):
        max_lv = 0 if not level_total else max(level_total.keys())
        results = {
            "hierarchy": hname,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluate_type": self.evaluate_type,
            "embedding_key": self.embedding_key or "none",
            "query_type": self.query_type if hasattr(self, 'query_type') and self.evaluate_type == "question" else "none",
            "run_number": run_number,
            "total_tested": tested_count
        }

        aggregate_score = sum(level_correct.values())
        max_possible_score = sum(level_total.values())
        aggregate_accuracy = aggregate_score / max_possible_score if max_possible_score > 0 else 0

        for lv in sorted(level_total.keys()):
            t_ = level_total[lv]
            if t_ > 0:
                c_ = level_correct[lv]
                results[f"level_{lv}_accuracy"] = f"{c_/t_:.4f}"
                results[f"level_{lv}_correct"] = c_
                results[f"level_{lv}_total"] = t_
            else:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = 0
                results[f"level_{lv}_total"] = 0

        results["aggregate_score"] = aggregate_score
        results["max_possible_score"] = max_possible_score
        results["aggregate_accuracy"] = f"{aggregate_accuracy:.4f}"

        file_exists = os.path.exists(out_csv)

        existing_max_lv = 0
        if file_exists:
            try:
                with open(out_csv, 'r', newline='') as fh:
                    reader = csv.DictReader(fh)
                    if reader.fieldnames:
                        for field in reader.fieldnames:
                            if field.startswith("level_") and field.endswith("_accuracy"):
                                level_num = int(field.split("_")[1])
                                existing_max_lv = max(existing_max_lv, level_num)
            except Exception:
                existing_max_lv = 0

        final_max_lv = max(max_lv, existing_max_lv)

        complete_fieldnames = ["hierarchy", "datetime", "evaluate_type", "embedding_key", "query_type", "run_number", "total_tested"]
        for lvv in range(final_max_lv + 1):
            complete_fieldnames.extend([
                f"level_{lvv}_accuracy",
                f"level_{lvv}_correct",
                f"level_{lvv}_total"
            ])
        complete_fieldnames.extend(["aggregate_score", "max_possible_score", "aggregate_accuracy"])

        for lv in range(final_max_lv + 1):
            if lv not in level_total:
                results[f"level_{lv}_accuracy"] = "none"
                results[f"level_{lv}_correct"] = "none"
                results[f"level_{lv}_total"] = "none"

        if file_exists:
            with open(out_csv, 'r', newline='') as fh:
                reader = csv.DictReader(fh)
                existing_fieldnames = list(reader.fieldnames) if reader.fieldnames else []

                if set(complete_fieldnames) - set(existing_fieldnames):
                    print(f"Adding new columns for level(s) up to {final_max_lv}")

                    temp_csv = out_csv + ".temp"
                    with open(out_csv, 'r', newline='') as fin, open(temp_csv, 'w', newline='') as fout:
                        reader = csv.DictReader(fin)
                        writer = csv.DictWriter(fout, fieldnames=complete_fieldnames)
                        writer.writeheader()

                        for row in reader:
                            for field in complete_fieldnames:
                                if field not in row:
                                    row[field] = "none"
                            writer.writerow(row)

                    os.replace(temp_csv, out_csv)
                    fieldnames = complete_fieldnames
                else:
                    fieldnames = existing_fieldnames
        else:
            fieldnames = complete_fieldnames

        with open(out_csv, 'a', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(results)
