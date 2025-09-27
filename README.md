# manchu-in-context-mt

### Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu: [ACL anthology link](https://aclanthology.org/2025.acl-long.429.pdf)  
  

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
@inproceedings{pei-etal-2025-understanding,
    title = "Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on {M}anchu",
    author = "Pei, Renhao  and
      Liu, Yihong  and
      Lin, Peiqin  and
      Yvon, Fran{\c{c}}ois  and
      Schuetze, Hinrich",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.429/",
    doi = "10.18653/v1/2025.acl-long.429",
    pages = "8767--8788",
    ISBN = "979-8-89176-251-0",
    abstract = "In-context machine translation (MT) with large language models (LLMs) is a promising approach for low-resource MT, as it can readily take advantage of linguistic resources such as grammar books and dictionaries.Such resources are usually selectively integrated into the prompt so that LLMs can directly perform translation without any specific training, via their in-context learning capability (ICL).However, the relative importance of each type of resource, e.g., dictionary, grammar book, and retrieved parallel examples, is not entirely clear.To address this gap, this study systematically investigates how each resource and its quality affect the translation performance, with the Manchu language as our case study. To remove any prior knowledge of Manchu encoded in the LLM parameters and single out the effect of ICL, we also experiment with an enciphered version of Manchu texts.Our results indicate that high-quality dictionaries and good parallel examples are very helpful, while grammars hardly help.In a follow-up study, we showcase a promising application of in-context MT: parallel data augmentation as a way to bootstrap a conventional MT model. When monolingual data abound, generating synthetic parallel data through in-context MT offers a pathway to mitigate data scarcity and build effective and efficient low-resource neural MT systems."
}
```
