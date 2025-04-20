fLMSci - LLM Based scientific hierarchography creation pipeline

Script | Pipeline type | Main steps
run_par.sh | Parallel | 1. Generate topics & rationales → 2. Place topics in parallel → 3. Merge chunked taxonomy → 4. Map papers → (optional) Evaluate
run_incr.sh | Incremental | 1. Generate topics & rationales → 2. Incrementally place each topic → 3. Map papers → (optional) Evaluate

To achieve the above pipelines we need to place json files inside the jsons folder first. Optionally, can also run each pipeline separately step by step by following their individual README files.

1. Give the shell scripts execute permission once
```bash
 chmod +x run_par.sh run_incr.sh
```

2. Parallel pipeline
```bash
 bash run_par.sh                # basic run
 bash run_par.sh --evaluate     # run + evaluation
```

3. Incremental pipeline
```bash
 bash run_incr.sh               # basic run
 bash run_incr.sh --evaluate    # run + evaluation
```
Optionally, we can also run it like:
```bash
 bash run_incr.sh --batch_size 16 --max_depth 8 --evaluate
```