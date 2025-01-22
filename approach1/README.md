In approach 1, we start with about 2190 papers and generate summary for each in summary_topics.py file and cluster them into 438 (2190/5) clusters and keep going until we have 17 clusters. 
2190 -> 438 -> 87 -> 17

First, download the data using the following link - [dataset](https://livejohnshopkins-my.sharepoint.com/:f:/r/personal/tnayak2_jh_edu/Documents/cagias?csf=1&web=1&e=LrepF3)

Now, let's load the OpenAI API key:
1. Create a .env file and add the below line to it:
```bash
OPENAI_API_KEY=your-api-key
```

2. ```bash
pip install python-dotenv
```


Based on the above downloaded dataset do update the path for input_folder in summary_topics.py

Run each of the files in sequence given below:
1. summary_topics.py
2. clustering_level1.py
3. clustering_87.py
4. clustering_12.py

At various intervals, the embeddings will be stored into the embeddings folder.

The results for each of the round of clustering will be stored under results folder.