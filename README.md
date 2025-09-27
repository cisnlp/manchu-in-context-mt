# manchu-in-context-mt

### Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu: [Arxiv link](https://arxiv.org/pdf/2502.11862)  
  

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
Run the following command to use Llama3_3B with the π(μ(x), Dl+s+c, Pbm) setting. This configuration uses the default dictionary variant Dl+s+c, retrieves parallel examples via BM25, and excludes the Grammar and CoT components.
```bash
python pipeline.py --model_id llama3_3b --test_sens test_sens337_mnc.txt --para bm25 --grammar None --cot None
```
The output is a list of tuples (mnc_sen,prompt,generated_text,translation), saved as `results_test_sens337_mnc_llama3_3b_bm25_None_None.pkl`.

## Generating prompts

Alternatively, you can use `generate_prompts.py` to only generate a list of prompts, without running the LLMs:
```bash
python generate_prompts.py --test_sens test_sens337_mnc.txt --para bm25 --grammar grammar_basic --cot None
```

## Citation
If you find our code useful for your research, please cite:

```
@misc{pei2025understandingincontextmachinetranslation,
      title={Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu}, 
      author={Renhao Pei and Yihong Liu and Peiqin Lin and François Yvon and Hinrich Schütze},
      year={2025},
      eprint={2502.11862},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.11862}, 
}
```
