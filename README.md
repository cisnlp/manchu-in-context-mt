# manchu-in-context-mt

### Installation

```bash
pip install -r requirements.txt
```

### Running manchu-in-context-mt

`pipeline.py` is the main script, it has the following arguments:

- `--model_id_short`: A shorthand identifier for selecting the LLM model. To view the available models or add new ones, see: `pipeline.py`.
- `--test_sens`: Path to the file containing test sentences, such as `test_sens337_mnc.txt`.

### Example
```bash
python pipeline.py --model_id_short llama3_3b --test_sens test_sens337_mnc.txt
```