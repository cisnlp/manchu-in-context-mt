from tqdm import tqdm
import argparse
import pickle
from pipeline.prompts import extract_enclosed_text, prompt_system, prompt_direct_translate, prompt_direct_translate_morph, prompt_template, component_cot_Ca, component_cot_Cas, component_para, component_grammar
def default_value():
    return 'not found in dictionary, could be a proper noun or a typo'
from pipeline.get_components import (convert2buleku_ortho, morphology_analyzer_plus_assembled_dict,get_manchu_entries,assembled_dict,
                            get_parallelSent, get_parallelSent_bm25_top_n,prepare_bm25_retrieval_corpus,
                            get_grammar_sections, grammar_sections_sp_dict,grammar_sections_l_dict,grammar_sections_lp_dict,
                            get_manchu_entries_encrypted, transform_sen, encrypt_parallelSent,
                            word_to_sentence_mnc2eng)


# Function to get arguments
def get_parser():
    parser = argparse.ArgumentParser(description="setting selection")
    # positional argument for test_sens, expecting a file path
    parser.add_argument("--test_sens", type=str, help="Path to the test sentences file")
    # positional argument for the setting for each component
    parser.add_argument("--para", type=str, help="Choose a variant for parallel sentences")
    parser.add_argument("--grammar", type=str, help="Choose a variant for grammar")
    parser.add_argument("--cot", type=str, help="Choose a variant for CoT prompting")
    return parser

if __name__ == '__main__':    
    # parse the cmd arguments e.g. python pipeline.py 1b test_sens1795.txt
    parser = get_parser() 
    args = parser.parse_args()  # Parse arguments from command line
    # get the input sentences
    mnc_sens_source = []
    with open(args.test_sens, mode='r',encoding='utf8') as file:
        for line in file:
            if len(line.strip().split('\t')) == 2:
                mnc_sen, source = line.strip().split('\t')
                mnc_sens_source.append((mnc_sen,source))
            else:
                mnc_sen = line.strip()
                source = 'UNKNOWN'
                mnc_sens_source.append((mnc_sen,source))
    
    # load the setting of components
    P = args.para
    G = args.grammar
    C = args.cot

    # prepare the corpus for bm25 retrieval
    bm25,tokenized_corpus,parallel_example_dict = prepare_bm25_retrieval_corpus(word_to_sentence_mnc2eng)

    # generate prompts
    prompts = []

    progress_bar_num = 50
    interval_num = (len(mnc_sens_source) // progress_bar_num) # interval for updating the bar
    with tqdm(total=progress_bar_num) as progress_bar:# update the bar for every interval_num sentences
        for i, sen_source in enumerate(mnc_sens_source):
            mnc_sen = sen_source[0]
            # components
            sent = convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_sen,assembled_dict))
            wordbyword = '\n'.join(get_manchu_entries(mnc_sen,assembled_dict,collocations=True,suffixes=True,masked_out=False))
            parallel_sentences = {
                "None": '',
                "bm25": component_para('Manchu', 'English','\n'.join(get_parallelSent_bm25_top_n(sent,bm25,tokenized_corpus,parallel_example_dict,n=10))),
                "dict": component_para('Manchu', 'English','\n'.join(get_parallelSent(mnc_sen,word_to_sentence_mnc2eng,assembled_dict)))
                }        
            grammar = {
                "None": '',
                "grammar_basic":component_grammar(''),
                "grammar_short":component_grammar(get_grammar_sections(mnc_sen,grammar_sections_sp_dict,assembled_dict)),
                "grammar_long":component_grammar(get_grammar_sections(mnc_sen,grammar_sections_l_dict,assembled_dict)),
                "grammar_long_para":component_grammar(get_grammar_sections(mnc_sen,grammar_sections_lp_dict,assembled_dict))
            }

            cot = {
                "None": '',
                "annotate":component_cot_Ca('Manchu', 'English'),
                "annotate_syntax":component_cot_Cas('Manchu', 'English')
            }

            # # encrypted components
            # sent_encrypted = transform_sen(convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_sen)))
            # wordbyword_encrypted = '\n'.join(get_manchu_entries_encrypted(mnc_sen,assembled_dict))
            # parallel_sentences_encrypted = component_para('Unknown language', 'English','\n'.join(encrypt_parallelSent(get_parallelSent_bm25_top_n(sent,bm5,tokenized_corpus,mnc_eng_parallel_example_dict,n=10))))
            # cot_encrypted = component_cot_Ca('Unknown language', 'English')
            # prompt = prompt_template('Manchu', 'English', sent_encrypted, wordbyword_encrypted, [parallel_sentences_encrypted,cot_encrypted])

            # e.g. π(μ(x), Dl+s+c, Pbm)
            prompt = prompt_template('Manchu', 'English', sent, wordbyword, [parallel_sentences[P],grammar[G],cot[C]])
            message = [
                {"role": "system", "content": prompt_system('English')},
                {"role": "user", "content": prompt}
            ]
            prompts.append(prompt)
            if i % interval_num == 0:   # update the progress_bar at every interval_num
                progress_bar.update(1)

    sens_prompts = [(sen_srouce[0],sen_srouce[1],prompt) for sen_srouce,prompt in zip(mnc_sens_source,prompts)]

    # Save the list to a pickle file
    with open(f"prompts_{args.test_sens.replace('.txt','')}_{P}_{G}_{C}.pkl", "wb") as f:
        pickle.dump(sens_prompts, f)
        print('prompts saved')