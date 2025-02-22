import re
import argparse
from huggingface_hub import login
from vllm import LLM, SamplingParams
import pickle
from pipeline.prompts import extract_enclosed_text, prompt_system, prompt_direct_translate, prompt_direct_translate_morph, prompt_template, component_cot_Ca, component_cot_Cas, component_para, component_grammar
from manchu_morphology_analyzer import manchu_morphology_analyzer
def default_value():
    return 'not found in dictionary, could be a proper noun or a typo'
from pipeline.get_components import (convert2buleku_ortho, morphology_analyzer_plus_assembled_dict,get_manchu_entries,
                            mnc_eng_parallel_example_dict, tokenized_corpus, get_parallelSent, get_parallelSent_bm25_top_n, 
                            get_grammar_sections, grammar_sections_lp_dict,
                            get_manchu_entries_encrypted, transform_sen, encrypt_parallelSent)

# Dictionary to map shorthand model names to full model IDs
MODEL_MAP = {
    "llama3_70b": "meta-llama/Llama-3.1-70B-Instruct",
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3_3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3_1b": "meta-llama/Llama-3.2-1B-Instruct"
    }

# Function to get arguments
def get_parser():
    parser = argparse.ArgumentParser(description="LLM model selection")
    # positional argument for model_id_short (expecting the shorthand code)
    parser.add_argument("--model_id_short", type=str, choices=MODEL_MAP.keys(),
                        help="Shorthand model ID (e.g., 'llama3_1b' for 'meta-llama/Llama-3.2-1B-Instruct')")
    # positional argument for test_sens, expecting a file path
    parser.add_argument("--test_sens", type=str, help="Path to the test sentences file")
    return parser

if __name__ == '__main__':
    # Prompt user to enter their Hugging Face token
    token = input("Please enter your Hugging Face token: ")
    login(token=token)
    
    # parse the cmd arguments e.g. python pipeline.py 1b test_sens1795.txt
    parser = get_parser() 
    args = parser.parse_args()  # Parse arguments from command line
    # get the input sentences
    mnc_sens = []
    with open(args.test_sens, mode='r',encoding='utf8') as file:
        for mnc_sen in file:
            mnc_sens.append(mnc_sen)
        
    # select a model
    model_id = MODEL_MAP[args.model_id_short]# Map the shorthand model_id to the full model ID
    
    llm = LLM(model=model_id, dtype='float16', 
            download_dir='./model_cache',
            max_model_len=20000)
        
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    # # offloading parts of the model due to hardware limitation
    # llm = LLM(model=model_id, dtype='float16', cpu_offload_gb=4.5, max_model_len=2000)

    # generate prompts
    prompt_messages = []
    for mnc_sen in mnc_sens:
        # components
        sent = convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_sen))
        wordbyword = '\n'.join(get_manchu_entries(mnc_sen,collocations=True,suffixes=True,masked_out=False))
        
        parallel_sentences = component_para('Manchu', 'English','\n'.join(get_parallelSent_bm25_top_n(sent,tokenized_corpus,mnc_eng_parallel_example_dict,n=10)))
        # grammar_basic = component_grammar('')
        # grammar_long = component_grammar(get_grammar_sections(mnc_sen,grammar_sections_l_dict))
        grammar_long_para = component_grammar(get_grammar_sections(mnc_sen,grammar_sections_lp_dict))
        # grammar_short = component_grammar(get_grammar_sections(mnc_sen,grammar_sections_sp_dict))
        # cot = component_cot('Manchu', 'English')

        # # encrypted components
        # sent_encrypted = transform_sen(convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_sen)))
        # wordbyword_encrypted = '\n'.join(get_manchu_entries_encrypted(mnc_sen))
        # parallel_sentences_encrypted = component_para('Unknown language', 'English','\n'.join(encrypt_parallelSent(get_parallelSent_bm25_top_n(sent,tokenized_corpus,mnc_eng_parallel_example_dict,n=10))))
        # cot_encrypted = component_cot('Unknown language', 'English')

        # π(μ(x), Dl+s+c, Pbm, Glp)
        prompt_Glp = prompt_template('Manchu', 'English', sent, wordbyword, [parallel_sentences,grammar_long_para])
        message = [
            {"role": "system", "content": prompt_system('Manchu')},
            {"role": "user", "content": prompt_Glp}
        ]
        prompt_messages.append(message)


    # Create a sampling params object.
    #sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
    sampling_params = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=5000)
    # Generate texts from the prompts, the outputs is a list of RequestOutput objects that contain the prompt, generated text, and other information.
    # #outputs = llm.generate(prompts, sampling_params,use_tqdm=True)
    outputs = llm.chat(prompt_messages,sampling_params=sampling_params,use_tqdm=True)

    # format the results  
    results = []
    for mnc_sen,output in zip(mnc_sens,outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        translation = extract_enclosed_text(generated_text)
        results.append((mnc_sen,prompt,generated_text,translation))
        # print(translation)

    # Save the list to a pickle file
    with open(f"results_{args.test_sens.replace('.txt','')}_{args.model_id_short}.pkl", "wb") as f:
        pickle.dump(results, f)
        print('results saved')