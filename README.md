# Science Cartography 🎨

The goal of this project is to develop interpretable, hierarchical representation of science papers. 

## JSON Structure
```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "title": "Level 3 Cluster Name",
      "abstract": "Level 3 Cluster Summary",
      "children": [
        {
          "cluster_id": 10,
          "title": "Level 2 Cluster Name",
          "abstract": "Level 2 Cluster Summary",
          "children": [
            {
              "cluster_id": 100,
              "title": "Level 1 Cluster Name",
              "abstract": "Level 1 Cluster Summary",
              "children": [
                {
                  "paper_id": 1000,
                  "title": "Paper Title",
                  "abstract": "Paper Abstract"
                }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

## The Visualization Tool 
By the visualization, you can:

- Navigate through different hierarchy .json file
- View cluster statistics
- Explore clusters / papers titles and abstracts

The hierarchy .json files are in the /hierarchies directory, each file named in the format of: `DATE_ALG_HYPERPARAMS_PAPERCOUNT.json`  
For example: `2025-01-13_k-means_gpt4-naming_k=5_1900.json`


### Setting up the Visualization Setup
Make sure you have Streamlit and other required dependencies.
```shell 
pip install streamlit pandas numpy
```
To run the visualization, run the code:
```shell 
streamlit run app.py
```
