# 🎨 SCIENCE HIERARCHOGRAPHY: Hierarchical Organization of Science Literature

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)

A tool for automatically generating hierarchical structures from scientific paper collections using:
1. Embeddings clustering techniques
2. LLM intelligence

The goal of this project is to develop interpretable, hierarchical representation of science papers.

## 📋 Table of Contents
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Approaches](#approaches)
  - [SciChic Hierarchy Generation](#scichic-hierarchy-generation)
  - [fLMSci Pipeline](#flmsci-pipeline)
- [Parameters Explanation](#parameters-explanation)

## 💡 Requirements
The requirements are listed in the `requirements.txt`. Use the following commands to build the environment for this project:

```bash
conda create -n science python=3.8
conda activate science
pip install -r requirements.txt
```

## 🗂️ Data Preparation
We have two paper collections available:
- The 2k paper collection **SciPile**
- The 10k paper collection **SciPileLarge**

You can use the following command to download:
```bash
cd download/
TODO
```

## 🔬 Approaches

### 🔮 SciChic Hierarchy Generation
The process has two main steps:

#### Generate Embeddings
First, make sure you have generated all the embeddings for your papers using:

```bash
python generate.py --input_folder /path/to/your/papers --output_file ./embeddings/your_embedding_name.pkl
```

#### Create Hierarchy
Then you can start creating the hierarchy with:

```bash
python main.py \
  --embedding_generator qwen \
  --summary_generator llama \
  --clustering_method kmeans \
  --evaluator qwen \
  --clustering_direction top_down \
  --base_path /project/directory/ \
  --cluster_sizes 276 40 6 \
  --run_time 1 \
  --evaluate_time 1 \
  --test_count 5 \
  --pre_generated_embeddings_file ./embedding_file.pkl \
  --evaluate_type normal \
  --embedding_source all
```

#### Parameters Explanation
- **embedding_generator**: Model used to generate embeddings (options: qwen, llama, etc.)
- **summary_generator**: Model used to generate summaries for clusters
- **clustering_method**: Algorithm for clustering (options: kmeans, hierarchical, etc.)
- **clustering_direction**: Direction of hierarchy building (top_down or bottom_up)
- **cluster_sizes**: Number of clusters at each level of the hierarchy
- **embedding_source**: Contribution type used to create the hierarchy:
  - **all**: Use all paper content
  - **problem**: Focus on problem statements
  - **solution**: Focus on proposed solutions
  - **results**: Focus on research results

### 🧵 fLMSci Pipeline
fLMSci is an LLM-based scientific hierarchography creation pipeline that offers two approaches:

#### Pipeline Types

| Script      | Pipeline type | Main steps                                                   |
| ----------- | ------------- | ------------------------------------------------------------ |
| run_par.sh  | Parallel      | 1. Generate topics & rationales → 2. Place topics in parallel → 3. Merge chunked taxonomy → 4. Map papers → (optional) Evaluate |
| run_incr.sh | Incremental   | 1. Generate topics & rationales → 2. Incrementally place each topic → 3. Map papers → (optional) Evaluate |

#### Setup & Execution

Before running the pipelines, you need to:
1. Place JSON files inside the `jsons` folder
2. Give the shell scripts execute permission (one-time step):
   ```bash
   chmod +x run_par.sh run_incr.sh
   ```

#### Running the Parallel Pipeline
```bash
bash run_par.sh                # basic run
bash run_par.sh --evaluate     # run + evaluation
```

#### Running the Incremental Pipeline
```bash
bash run_incr.sh               # basic run
bash run_incr.sh --evaluate    # run + evaluation
```

You can also customize the run with additional parameters:
```bash
bash run_incr.sh --batch_size 16 --max_depth 8 --evaluate
```

Note: Each pipeline can also be run step by step by following their individual README files.