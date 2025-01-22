import json
import os
import openai
import pandas as pd
import re
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = openai_api_key


def extract_title_and_abstract_from_folder(input_folder, output_folder):
    """
    Extracts titles and abstracts from JSON files in the input folder 
    and writes them to text files in the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)
    missing_files = 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(input_folder, file_name)
            output_file_name = file_name.replace('.json', '.txt')
            output_file_path = os.path.join(output_folder, output_file_name)

            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                try:
                    data = json.load(json_file)
                    if 'title' in data and 'abstract' in data:
                        title = data['title']
                        abstract = data['abstract']

                        with open(output_file_path, 'w', encoding='utf-8') as output_file:
                            output_file.write(f"Title: {title}\n")
                            output_file.write(f"Abstract: {abstract}\n\n")
                    else:
                        missing_files += 1
                        print(f"Skipping {file_name}: Missing 'title' or 'abstract'.")

                except json.JSONDecodeError:
                    missing_files += 1
                    print(f"Error decoding JSON in file: {file_name}")

    print(f"Extraction complete. {missing_files} files skipped due to missing 'title' or 'abstract' fields.")

def process_research_papers(input_folder, model="gpt-4o", user_prompt=""):
    """
    Processes research papers from text files in the input folder, extracts their titles and abstracts,
    generates summaries using the OpenAI API, and saves the results to a CSV file.
    """
    file_names = []
    titles = []
    abstracts = []
    full_outputs = []
    file_count = 0
    skipped_files = 0

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.txt'):
            input_file_path = os.path.join(input_folder, file_name)

            with open(input_file_path, 'r', encoding='utf-8') as input_file:
                file_content = input_file.read()

            lines = file_content.split("\n")
            title = lines[0].strip().replace("Title:", "").strip() if len(lines) > 0 else ""
            abstract = "\n".join(lines[1:]).strip().replace("Abstract:", "").strip() if len(lines) > 1 else ""

            if not title or not abstract:
                skipped_files += 1
                print(f"Skipping file due to missing title or abstract: {file_name}")
                continue

            system_prompt = "You are an experienced scientist who is going to read and review research papers."
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Title: {title}\nAbstract: {abstract}\n{user_prompt}"}
            ]

            try:
                if file_count % 5 == 0:
                    print(f"Processed {file_count} files.")
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages
                )
                full_output = response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Error generating summary for file {file_name}: {e}")
                continue

            file_names.append(file_name)
            titles.append(title)
            abstracts.append(abstract)
            full_outputs.append(full_output)
            file_count += 1

    print(f"Processed {file_count} files. Skipped {skipped_files} files.")

    data = {
        "File Name": file_names,
        "Title": titles,
        "Abstract": abstracts,
        "json_output": full_outputs
    }
    df = pd.DataFrame(data)
    print('Dataframe created')
    return df

def extract_json_output(json_output):
    match = re.search(r'```json(.*?)```', json_output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def parse_json_column(df, json_column):
    # Lists to store parsed data
    summaries = []
    one_sentence_summaries = []
    topics_list = []
    rationales_list = []

    # Iterate through each JSON string in the specified column
    for json_str in df[json_column]:
        try:
            # Parse the JSON string
            parsed_data = json.loads(json_str)
            
            # Extract components
            summaries.append(parsed_data.get("summary", ""))
            
            # Collect all topics and rationales into lists of lists
            topics = [topic.get("topic") for topic in parsed_data.get("topics", [])]
            rationales = [topic.get("rationale") for topic in parsed_data.get("topics", [])]
            
            topics_list.append(topics)
            rationales_list.append(rationales)
        except json.JSONDecodeError:
            # Handle any JSON parsing errors
            summaries.append("")
            one_sentence_summaries.append("")
            topics_list.append([])
            rationales_list.append([])

    # Add the parsed data as new columns to the DataFrame
    df["Summary"] = summaries
    df["Topics"] = topics_list
    df["Rationales"] = rationales_list

    return df


# Define paths
input_folder = 'YOUR_INPUT_FOLDER'
output_folder = 'approach1/abstracts'
prompt_path = 'approach1/prompts/summary_topics.txt'

# Load the user prompt
with open(prompt_path, 'r', encoding='utf-8') as file:
    user_prompt = file.read().strip()

# Run the pipeline
extract_title_and_abstract_from_folder(input_folder, output_folder)
df = process_research_papers(output_folder, user_prompt=user_prompt)

df['json_output'] = df['json_output'].apply(extract_json_output)
# Parse the JSON column and update the DataFrame
updated_df = parse_json_column(df, "json_output")

#drop the json_output column
updated_df = updated_df.drop(columns=["json_output"])

# save the updated dataframe to a new csv file
updated_df.to_csv('approach1/results/summaries_topics.csv', index=False)
print('Dataframe saved to CSV file.')
print('Processed summaries and topics saved to summaries_topics.csv')