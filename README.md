# Adversarial-factuality

Code from [`Core` paper](https://arxiv.org/abs/2407.03572). 

## Added Features
Modified for the following functionality:

- Load pre-decomposed passages with `PreComputedDecomposer`
- Verify claims against a "local" reference (i.e., no retrieval)
- Verify claims against MedRAG corpus for medical text
- Write out results in `JSONLines` instead of `JSON` format
- Dynamic data loading with customizable `topic_key`, `src_key`, and `text_key` for loading different datasets

## Repo Setup
The codebase is very customizable. A typical decompose-then-verify workflow is:

1. Load data in a `ScoreGenerationTask`
2. Decompose each passage with a `Decomposer`. This may entail an LLM call.
3. Verify with a `LLMSupportScorer`



## Usage

The pipeline depends on API calls for the decomposition and verification steps. 

1. Hosting a local model with vLLM. This example assumes deployment on a cluster with SLURM scheduler.
```shell
#!/bin/bash
#SBATCH --job-name=serve-vllm
#SBATCH --time=12:00:00
#SBATCH --partition=ica100,a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30GB
#SBATCH -A <account name>
#SBATCH --output=logs/%x.%j.log

module load anaconda
conda activate <environment with Core requirements>

SLURM_JOB_GPUS=1  # Set this to --gres=gpu:1
CUDA_VISIBLE_DEVICES="0"
for i in $(seq 1 "$(( SLURM_JOB_GPUS - 1 ))"); do
  CUDA_VISIBLE_DEVICES+=",${i}"
done

export CUDA_VISIBLE_DEVICES
export PORT=22659
export MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"

echo "SLURM ${SLURM_JOB_GPUS}"
echo "CUDA $CUDA_VISIBLE_DEVICES"

vllm serve "${MODEL_NAME}" \
    --dtype auto \
    --tensor-parallel-size "${SLURM_JOB_GPUS}" \
    --seed 42 \
    --tokenizer-mode "auto" \
    --uvicorn-log-level "warning" \
    --port ${PORT}
```


2. Launching Core. See below for example configurations.

```shellscript
python scripts/run_task.py CONFIG_PATH [--cache-path CACHE_PATH]
```


### Configurations

**FActScore**

```yaml
task:
  type: "score-generation"
  text_key: "${TEXT_KEY}"
  source_key: "${REF_KEY}"
  topic_key: "${ID_KEY}"
  scorer:
    type: "decompose"
    abstention_detector: "factscore"
    decomposer:
      type: "factscore"
      model_name: "${MODEL_NAME}"
      base_url: "${SERVER}/v1"
      api_key: "${API_KEY}"
      example_path: "${DECOMP_EXAMPLES}"
      sentencize: True
    base_scorer:
      type: "llm-support-local"
      model_name: "${MODEL_NAME}"
      base_url: "${SERVER}/v1"
      api_key: "${API_KEY}"
    aggregator:
      type: "factscore"
      gamma: 0
  input_path: "${INPUT_PATH}"
  output_path: "${OUTPUT_PATH}"
```

**Core**
```yaml
task:
  type: "score-generation"
  text_key: "${TEXT_KEY}"
  source_key: "${REF_KEY}"
  topic_key: "${ID_KEY}"
  scorer:
    type: "decompose"
    abstention_detector: "factscore"
    decomposer:
      type: "deduplicated"
      base_decomposer:
        type: "factscore"
        model_name: "${MODEL_NAME}"
        base_url: "${SERVER}/v1"
        api_key: "${API_KEY}"
        example_path: "${DECOMP_EXAMPLES}"
        sentencize: False
      sentence_level_checkworthy_scorer:
        type: "llm-checkworthy-general"
        model_name: "${MODEL_NAME}"
        base_url: "${SERVER}/v1"
        api_key: "${API_KEY}"
        in_batch_num: 8
      claim_level_checkworthy_scorer:
        type: "unli-confidence-boost-from-file"
        bleached_templates_path: "${BLEACHED_TEMPLATES}"
        entailer:
          type: "soft-entailer"
          model_name: "Zhengping/roberta-large-unli"
          device: "cuda"
          max_length: 256
          internal_batch_size: 32
          cache_dir: "${CACHE_DIR}"
        cap_entailer:
          type: "default"
          model_name: "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
          device: "cuda"
          max_length: 256
          internal_batch_size: 512
          cache_dir: "${CACHE_DIR}"
      entailer:
        type: "default"
        model_name: "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        device: "cuda"
        max_length: 256
        internal_batch_size: 512
        cache_dir: "${CACHE_DIR}"
    base_scorer:
      type: "llm-support-local"
      model_name: "${MODEL_NAME}"
      base_url: "${SERVER}/v1"
      api_key: "${API_KEY}"
    aggregator:
      type: "factscore"
      gamma: 0
  input_path: "${INPUT_PATH}"
  output_path: "${OUTPUT_PATH}"
```

**Pre-Computed claims**
```yaml
task:
  type: "score-generation"
  text_key: "${TEXT_KEY}"
  source_key: "${REF_KEY}"
  topic_key: "${ID_KEY}"
  scorer:
    type: "decompose"
    abstention_detector: "factscore"
    decomposer:
      type: "precomputed_decomposer"
      claims_path: "${CLAIMS_PATH}"
      topic_key: "${ID_KEY}"
      claims_key: "${CLAIMS_KEY}"
    base_scorer:
      type: "llm-support-local"
      model_name: "${MODEL_NAME}"
      base_url: "${SERVER}/v1"
      api_key: "${API_KEY}"
    aggregator:
      type: "factscore"
      gamma: 0
  input_path: "${INPUT_PATH}"
  output_path: "${OUTPUT_PATH}"
```

### Dataset for Sampling Generation

Since there were only ~200 datapoints in the FActScore dataset, we run the filtering over all the data from [nkandpa2/pretraining_entities](https://huggingface.co/datasets/nkandpa2/pretraining_entities) as well as the [popQA](https://github.com/AlexTMallen/adaptive-retrieval), aligning popularity metrics from both sources using Wikidata entity ids, resulting in the following dump format:

```json
[
    {
        "entity_id": "Q189729", // entity id of wikidata
        "wikidata_freq": 1294, // number of entity mention from wikidata
        "popqa_freq": 45173, // popularity indicator from popQA
        "popqa_entity_name": "Philip Glass", // entity name from popQA
        "wikipedia_page": "https://en.wikipedia.org/wiki/Philip_Glass", // wikipedia page
        "wikipedia_title": "Philip_Glass", // wikipedia title
        "is_in_dump": true, // whether this entity can be queried in the FActScore-provided wikipedia dump
        "adjusted_freq": 45173, // adjusted frequency, max(popqa_freq, wikidata_freq)
        "adjusted_freq_source": "popqa", // either "popqa" or "wikidata", the source of adjusted frequency
        "already_selected": false // whether this entity has been selected for the FActScore dataset
    },
    // ...
]
```

### Gotchas

#### Why is the batchified result different?

One thing that leads to inconsistency is that we use batched queries for the checkworthiness. This might leads to different queries being batched together, and thus the result might be different. To mitigate this, try to set `in_batch_num = 1` for the `sentence_level_checkworthy_scorer` in the config.
