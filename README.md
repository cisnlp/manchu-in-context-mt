# manchu-in-context-mt

## Installation

```bash
pip install -r requirements.txt
```

## Running manchu-in-context-mt

`pipeline.py` is the main script, it has the following arguments:

- `--model_id`: A shorthand identifier for selecting the LLM model. To view the available models or add new ones, see MODEL_MAP in `pipeline.py`.
- `--test_sens`: Path to the file containing test sentences, such as `test_sens337_mnc.txt`.
- `--para`: Select the variant for the Parallel Examples component from the following options: "None", "bm25", or "dict".
- `--grammar`: Select the variant for the Grammar component from the following options: "None", "grammar_basic", "grammar_short", "grammar_long", or "grammar_long_para".
- `--cot`: Select the variant for the CoT prompting component from the following options: "None", "annotate", "annotate_syntax".

### Example
```bash
python pipeline.py --model_id llama3_3b --test_sens test_sens337_mnc.txt --para bm25 --grammar None --cot None
```