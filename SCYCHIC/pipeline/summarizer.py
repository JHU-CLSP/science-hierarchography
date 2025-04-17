"""
summarizer.py: Summarize clusters using a real Llama model from HuggingFace transformers.
"""

import torch
import transformers
import random
import json
import re
import os
import time
from pathlib import Path
from typing import Tuple, Optional, List
from openai import OpenAI

class LlamaSummaryGenerator:
    def __init__(self,
                 model_id="meta-llama/Llama-3.3-70B-Instruct",
                 prompt_file: Path=None,
                 device_map="auto"):
        """
        If prompt_file is given, we read that prompt template, else do a generic prompt.
        We'll load huggingface pipeline for real usage.
        """
        self.model_id = model_id
        self.prompt_file = prompt_file
        print(f"Initializing Llama model: {model_id}")

        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def cleanup(self):
        """清理模型资源"""
        print("Starting cleanup of LlamaSummaryGenerator...")
        
        if torch.cuda.is_available():
            before_clean = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory before cleanup: {before_clean:.2f}MB")

        if self.model:
            print("- Cleaning model...")
            self.model = None
            print("- Model cleaned")

        if self.tokenizer:
            print("- Cleaning tokenizer...")
            self.tokenizer = None
            print("- Tokenizer cleaned")

        if self.pipeline:
            print("- Cleaning pipeline...")
            self.pipeline = None
            print("- Pipeline cleaned")

        print("- Cleaning GPU cache...")
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            after_clean = torch.cuda.memory_allocated() / 1024**2
            print(f"GPU memory after cleanup: {after_clean:.2f}MB")
            print(f"Memory freed: {before_clean - after_clean:.2f}MB")

        print("LlamaSummaryGenerator cleanup completed")

    def generate_cluster_summary(self, texts, debug=True):
        """
        Return (cluster_name, cluster_summary) by actually running the model.
        We'll parse JSON from the output, handling both simple and complex structured outputs.
        """
        if not texts:
            return ("EmptyCluster", "No texts provided.")

        if self.prompt_file:
            with open(self.prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()

        cluster_content = "\n\n".join(texts)
        prompt_str = prompt_template % cluster_content

        outs = self.pipeline(prompt_str, max_new_tokens=300)
        gen_text = outs[0]["generated_text"][len(prompt_str) - 25:]

        start = gen_text.find('{')
        end = gen_text.rfind('}') + 1
        
        if start == -1 or end <= start:
            print(gen_text, flush=True)
            return "Error in Summarization", f"Failed to find valid JSON in output for cluster with {len(texts)} items"
            
        json_str = gen_text[start:end]
        
        try:
            data = json.loads(json_str)
            cluster_name = data.get("Cluster Name", "Unnamed Cluster")
            
            if "Cluster Summary" in data:
                cluster_summary = data["Cluster Summary"]
            elif any(field in data for field in ["Problem", "Solution", "Results"]):
                structured_data = {}
                for field in ["Problem", "Solution", "Results"]:
                    if field in data:
                        structured_data[field] = data[field]
                cluster_summary = json.dumps(structured_data, ensure_ascii=False, indent=2)
            else:
                data_copy = data.copy()
                if "Cluster Name" in data_copy:
                    data_copy.pop("Cluster Name")
                cluster_summary = json.dumps(data_copy, ensure_ascii=False, indent=2)
                
            if debug:
                print("\n[DEBUG Output]", cluster_name, cluster_summary, flush=True)
            return cluster_name, cluster_summary
        except Exception as e:
            print(f"Error parsing Llama output: {e}")
            return "Error in Summarization", f"Failed to summarize cluster with {len(texts)} items: {str(e)}"

class GPTSummaryGenerator:
    def __init__(self,
                model_id: str = "gpt-4o-2024-08-06",
                prompt_file: Optional[Path] = None,
                api_key: Optional[str] = None):
        """
        Initialize a GPT-based summary generator.
        
        Args:
            model_id: OpenAI model identifier
            prompt_file: Path to prompt template file
            api_key: OpenAI API key (if None, will be fetched from environment)
        """
        self.model_id = model_id
        self.prompt_file = prompt_file
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
        
        print(f"Initializing GPT summary generator with model: {model_id}")
        
        self.client = OpenAI(api_key=self.api_key)

    def generate_cluster_summary(self, texts: List[str], debug: bool = True) -> Tuple[str, str]:
        """
        Generate a summary for a cluster of texts.
        
        Args:
            texts: List of texts to summarize
            debug: Whether to print debug information
            
        Returns:
            Tuple of (cluster_name, cluster_summary)
        """
        if not texts:
            return ("EmptyCluster", "No texts provided.")

        if not self.prompt_file:
            raise ValueError("No prompt file provided")
        
        if not Path(self.prompt_file).exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
            
        with open(self.prompt_file, 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        cluster_content = "\n\n".join(texts)
        prompt_str = prompt_template % cluster_content
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": "You are a scientific research expert assistant."},
                        {"role": "user", "content": prompt_str}
                    ],
                    temperature=0,
                    max_tokens=500,
                    n=1,
                    timeout=60
                )
                
                if debug:
                    total_tokens = response.usage.total_tokens
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    
                    if self.model_id.startswith("gpt-4"):
                        input_cost_per_1k = 0.03
                        output_cost_per_1k = 0.06
                    else:
                        input_cost_per_1k = 0.0015
                        output_cost_per_1k = 0.002
                        
                    input_cost = (prompt_tokens / 1000) * input_cost_per_1k
                    output_cost = (completion_tokens / 1000) * output_cost_per_1k
                    total_cost = input_cost + output_cost
                    
                    print(f"[DEBUG] API call: {total_tokens} tokens (Input: {prompt_tokens}, Output: {completion_tokens})")
                    print(f"[DEBUG] Estimated cost: ${total_cost:.4f}")
                
                gen_text = response.choices[0].message.content
                
                if debug:
                    print("\n[DEBUG Output]", gen_text[:500] + ("..." if len(gen_text) > 500 else ""), flush=True)
                
                start = gen_text.find('{')
                end = gen_text.rfind('}') + 1
                
                if start == -1 or end <= start:
                    print("Error: Could not find valid JSON in response")
                    return "Error in Summarization", f"Failed to extract JSON from model output for cluster with {len(texts)} items"
                
                json_str = gen_text[start:end]
                
                data = json.loads(json_str)
                cluster_name = data.get("Cluster Name", "Unnamed Cluster")
                
                if "Cluster Summary" in data:
                    cluster_summary = data["Cluster Summary"]
                elif any(field in data for field in ["Problem", "Solution", "Results"]):
                    structured_data = {}
                    for field in ["Problem", "Solution", "Results"]:
                        if field in data:
                            structured_data[field] = data[field]
                    cluster_summary = json.dumps(structured_data, ensure_ascii=False, indent=2)
                else:
                    data_copy = data.copy()
                    if "Cluster Name" in data_copy:
                        data_copy.pop("Cluster Name")
                    cluster_summary = json.dumps(data_copy, ensure_ascii=False, indent=2)
                
                return cluster_name, cluster_summary
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from GPT output: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return "Error in Summarization", f"Failed to parse JSON from model output after {max_retries} attempts"
                    
            except Exception as e:
                print(f"Error in GPT API call: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    return "Error in Summarization", f"API error after {max_retries} attempts: {str(e)}"

    def cleanup(self):
        """
        Dummy cleanup method for compatibility with existing code that expects this method.
        For API-based models, no actual cleanup is necessary.
        """
        print("GPTSummaryGenerator cleanup called (no action needed for API-based models)")