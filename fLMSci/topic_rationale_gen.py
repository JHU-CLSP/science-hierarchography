import json
import os
import openai
import pandas as pd
import re
import argparse
from dotenv import load_dotenv
from typing import List
from openai import OpenAI


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = openai.api_key) 

def extract_title_and_abstract_from_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    missing_files = 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(input_folder, file_name)
            output_file_path = os.path.join(output_folder, file_name.replace('.json', '.txt'))

            with open(json_file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    title = data.get('title', '').strip()
                    abstract = data.get('abstract', '').strip()

                    if title and abstract:
                        with open(output_file_path, 'w', encoding='utf-8') as out:
                            out.write(f"Title: {title}\nAbstract: {abstract}\n\n")
                    else:
                        missing_files += 1
                        print(f"Skipping {file_name}: Missing title or abstract.")
                except json.JSONDecodeError:
                    missing_files += 1
                    print(f"Error decoding JSON in file: {file_name}")
    
    print(f"Extraction complete. {missing_files} files skipped.")

def process_research_papers(input_folder, model="gpt-4", user_prompt=""):
    file_names, titles, abstracts, outputs = [], [], [], []
    file_count, skipped = 0, 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            with open(os.path.join(input_folder, file_name), 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split("\n")
            title = lines[0].replace("Title:", "").strip() if lines else ""
            abstract = "\n".join(lines[1:]).replace("Abstract:", "").strip() if len(lines) > 1 else ""

            if not title or not abstract:
                skipped += 1
                print(f"Skipping {file_name}: Missing title or abstract.")
                continue

            messages = [
                {"role": "system", "content": "You are an experienced scientist who is going to review research papers."},
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}\n{user_prompt}"}
            ]

            try:
                if file_count % 5 == 0:
                    print(f"Processed {file_count} files.")
                response = client.responses.create(model=model, input=messages)
                result = response.output_text
            except Exception as e:
                print(f"Error for {file_name}: {e}")
                continue

            file_names.append(file_name)
            titles.append(title)
            abstracts.append(abstract)
            outputs.append(result)
            file_count += 1

    print(f"✅ Completed: {file_count} files, Skipped: {skipped}")
    return pd.DataFrame({
        "File Name": file_names,
        "Title": titles,
        "Abstract": abstracts,
        "json_output": outputs
    })

def extract_json_output(text):
    match = re.search(r'```json(.*?)```', text, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_json_column(df, json_column):
    topics_list, rationales_list = [], []

    for row in df[json_column]:
        try:
            parsed = json.loads(row)
            topics = [t.get("topic") for t in parsed.get("topics", [])]
            rationales = [t.get("rationale") for t in parsed.get("topics", [])]
            topics_list.append(topics)
            rationales_list.append(rationales)
        except Exception:
            topics_list.append([])
            rationales_list.append([])

    df["Topics"] = topics_list
    df["Rationales"] = rationales_list
    return df.drop(columns=[json_column], errors='ignore')

def save_unique_topics(df: pd.DataFrame, output_path: str):
    all_topics: List[str] = sum(df["Topics"].tolist(), [])
    unique_topics = sorted(set(topic for topic in all_topics if topic))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for topic in unique_topics:
            f.write(topic + "\n")
    
    print(f"📄 Saved {len(unique_topics)} unique topics to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Process research papers and extract topics/rationales.")
    parser.add_argument("--input_folder", required=True, help="Path to folder with raw JSON files")
    parser.add_argument("--output_folder", required=True, help="Folder to save extracted title + abstract TXT files")
    parser.add_argument("--prompt_path", required=True, help="Text file containing prompt for GPT")
    parser.add_argument("--output_csv", required=True, help="Path to save final output CSV")
    parser.add_argument("--model", default="gpt-4", help="OpenAI model to use (default: gpt-4)")
    parser.add_argument("--topics_txt", required=True, help="Path to save unique topics text file")
    args = parser.parse_args()

    with open(args.prompt_path, 'r', encoding='utf-8') as f:
        user_prompt = f.read().strip()

    extract_title_and_abstract_from_folder(args.input_folder, args.output_folder)
    df = process_research_papers(args.output_folder, model=args.model, user_prompt=user_prompt)
    df['json_output'] = df['json_output'].apply(extract_json_output)
    df = parse_json_column(df, "json_output")
    df.to_csv(args.output_csv, index=False)

    print(f"✅ Done! Topics + Rationales saved to: {args.output_csv}")
    save_unique_topics(df, args.topics_txt)
    print(f"📄 Unique topics saved to: {args.topics_txt}")
if __name__ == "__main__":
    main()

# Example usage:
# python your_script.py \
#   --input_folder abstracts \
#   --output_folder abstracts \
#   --prompt_path prompts/topic_gen.txt \
#   --output_csv results/topics_rationales.csv
#   --model gpt-4