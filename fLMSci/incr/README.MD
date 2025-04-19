This part of the repository deals with the fLMSci incremental approach in which we place units of information one by one by giving the LLM an action option at each point. 

1. To run this we need a seed taxonomy and units of information(list of topics). We can generate the list of unique topics by using fLMSci/par/code/topic_rationale_gen.py file.
2. Then just modify the example command below to run the code:
Example usage:
```bash
python fLMSci/incr/code/llm_incr.py \
     --topics_path path/to/unique_topics.txt \
     --taxonomy_path path/to/science_seed.json \
     --results_dir fLMSci/incr/results \
     --batch_size 32 \
     --max_depth 10 \
     --max_tokens 32000 \
     --max_response_tokens 256
```
By running this we will get a taxonomy of topics. We then map it to its respective papers using fLMSci/par/code/map_papers.py
