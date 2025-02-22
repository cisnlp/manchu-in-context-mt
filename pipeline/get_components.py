import os
import pickle, re, random
import yaml
from manchu_morphology_analyzer import manchu_morphology_analyzer
from rank_bm25 import BM25Okapi

def convert2buleku_ortho(w):
    #w = re.sub('=$','mbi',w)
    w = w.replace("g\'",'gg').replace("g\'",'gg').replace("k\'",'kk').replace("h\'",'hh').replace('c','q')# it seems converting transliteration on the fly can sometimes result in error, so convert them at the beginning before pickle them
    w = w.replace('dz','Z')
    w = w.replace('z','r')
    w = w.replace('Z','z')
    return w

# Get the absolute path of the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# load the pickled default dictionary
def default_value():
    return 'not found in dictionary, could be a proper noun or a typo'
with open(os.path.join(script_dir, "assembled_dict.pkl"), "rb") as f:  # 'rb' means read in binary mode
    assembled_dict = pickle.load(f)
with open(os.path.join(script_dir, "assembled_dict_encrypted.pkl"), "rb") as f:  # 'rb' means read in binary mode
    assembled_dict_encrypted = pickle.load(f)
# load the word_to_sentence dict, for retrieving parallel sentences for each word
with open(os.path.join(script_dir, "word_to_sentence_dict.pkl"), "rb") as f:
    word_to_sentence = pickle.load(f)
# load the freq_token_list dict, for experiment of masking out unfrequent token during retrieving dictionary entries
with open(os.path.join(script_dir, "freq_token_list.pkl"), "rb") as f:
    freq_token_list = pickle.load(f)
# load the yaml file containing grammar sections into dict
with open(os.path.join(script_dir, "grammar_sections_long.yaml"), "r", encoding="utf-8") as file:
    grammar_sections_l_dict = yaml.safe_load(file) #safe_load is recommended because it avoids executing potentially unsafe Python code embedded in the YAML file.
with open(os.path.join(script_dir, "grammar_sections_long_para.yaml"), "r", encoding="utf-8") as file:
    grammar_sections_lp_dict = yaml.safe_load(file) #safe_load is recommended because it avoids executing potentially unsafe Python code embedded in the YAML file.
with open(os.path.join(script_dir, "grammar_sections_short_para.yaml"), "r", encoding="utf-8") as file:
    grammar_sections_sp_dict = yaml.safe_load(file) #safe_load is recommended because it avoids executing potentially unsafe Python code embedded in the YAML file.

# create the corpus for bm25 retrieval of parallel examples
all_sen = [sen for sens in word_to_sentence.values() for sen in sens]
all_sen = list(set(all_sen))
# create a mnc:eng dict
mnc_eng_parallel_example_dict = {manchu_morphology_analyzer.noun_verb_splitter(mnc): eng.strip() for mnc, eng in (pair.split('\n') for pair in all_sen)}
# create the corpus for retrieval
corpus = list(mnc_eng_parallel_example_dict.keys())
# split each sentence according to morpheme
tokenized_corpus = [sen.replace(',',' ,').replace('.',' .').replace('=',' =').replace('~',' ~').split(' ') for sen in corpus]
# Initialize BM25
bm25 = BM25Okapi(tokenized_corpus)

# a dictionary that converts suffix into gloss
suffix_to_gloss_dict ={
    '~be':'ACCUSATIVE','be':'ACCUSATIVE',
    '~i':'GENITIVE','~ni':'GENITIVE','i':'GENITIVE','ni':'GENITIVE',
    '~de':'DATIVE','de':'DATIVE',
    '~qi':'ABLATIVE','qi':'ABLATIVE',
    '~deri':'PROLATIVE','deri':'PROLATIVE',
    '~ngga':'derivative suffix used to form nouns of quality','~ngge':'derivative suffix used to form nouns of quality','~nggo':'derivative suffix used to form nouns of quality',
    '~sa':'Plurals','~se':'Plurals','~so':'Plurals',
    '~si':'Plurals',
    '~ta':'Plurals','~te':'Plurals',
    '~ri':'Plurals',
    '=qi':'Conditional converb',
    '=qibe':'Concessive converb',
    '=qina':'Desiderative',
    '=fi':'Perfect converb',
    '=pi':'Perfect converb',
    '=mpi':'Perfect converb',
    '=ha':'Perfect participle','=he':'Perfect participle','=ho':'Perfect participle',
    '=ka':'Perfect participle','=ke':'Perfect participle','=ko':'Perfect participle',
    '=ngka':'Perfect participle','=ngke':'Perfect participle','=ngko':'Perfect participle',
    '=habi':'Perfect tense','=hebi':'Perfect tense','=hobi':'Perfect tense',
    '=kabi':'Perfect tense','=kebi':'Perfect tense','=kobi':'Perfect tense',
    '=ngkabi':'Perfect tense','=ngkebi':'Perfect tense','=ngkobi':'Perfect tense',
    '=hai':'Stative converb','=hei':'Stative converb','=hoi':'Stative converb',
    '=kai':'Stative converb','=kei':'Stative converb','=koi':'Stative converb',
    '=hakv':'Negative Participle Forms','=hekv':'Negative Participle Forms', '=hokv':'Negative Participle Forms',
    '=rakv':'Negative Participle Forms', '=rekv':'Negative Participle Forms', '=rokv':'Negative Participle Forms',
    '=hangge':'Substantive Forms of Participles', '=hengge':'Substantive Forms of Participles', '=hongge':'Substantive Forms of Participles',
    '=kangge':'Substantive Forms of Participles', '=kengge':'Substantive Forms of Participles', '=kongge':'Substantive Forms of Participles',
    '=ngkangge':'Substantive Forms of Participles', '=ngkengge':'Substantive Forms of Participles', '=ngkongge':'Substantive Forms of Participles',
    '=rangge':'Substantive Forms of Participles', '=rengge':'Substantive Forms of Participles', '=rongge':'Substantive Forms of Participles',
    '=darangge':'Substantive Forms of Participles', '=derengge':'Substantive Forms of Participles', '=dorongge':'Substantive Forms of Participles',
    '=ndarangge':'Substantive Forms of Participles', '=nderengge':'Substantive Forms of Participles', '=ndorongge':'Substantive Forms of Participles',
    '=ki':'Desiderative',
    '=kini':'Desiderative',
    '=mbi':'Aorist',
    '=mbihe':'Durative hypothetical imperfect tense',
    '=mbime':'Durative simultaneous converb',
    '=me':'Imperfect converb',
    '=nggala':'Prefatory converb', '=nggele':'Prefatory converb', '=nggolo':'Prefatory converb',    
    '=ra':'Imperfect participle', '=re':'Imperfect participle', '=ro':'Imperfect participle',
    '=dara':'Imperfect participle', '=dere':'Imperfect participle', '=doro':'Imperfect participle',
    '=ndara':'Imperfect participle', '=ndere':'Imperfect participle', '=ndoro':'Imperfect participle',
    '=rahv':'Temeritive apprehensive converb', '=rehv':'Temeritive apprehensive converb', '=rohv':'Temeritive apprehensive converb',
    '=tai':'adverbials', '=tei':'adverbials', '=toi':'adverbials',
    '=tala':'Terminative converb', '=tele':'Terminative converb', '=tolo':'Terminative converb'
}


# wrap the manchu_morphology_analyzer, before using manchu_morphology_analyzer, first check if the token is already in the entries of assembled_dict
def morphology_analyzer_plus_assembled_dict(sen):
    analysis_list = []
    for word in sen.split(' '):
        multiple_possible_analysis = set()
        # if the word is found in the entries of assembled_dict
        if assembled_dict[convert2buleku_ortho(word)] != 'not found in dictionary, could be a proper noun or a typo':
            multiple_possible_analysis.add(word)
        # then always go through the manchu_morphology_analyzer
        manchu_morphology_analyzer_analyses = manchu_morphology_analyzer.noun_verb_splitter(word).split('/')
        for analysis in manchu_morphology_analyzer_analyses:
            multiple_possible_analysis.add(analysis)
        analyzed_token = r'/'.join(multiple_possible_analysis)
        analysis_list.append(analyzed_token)
    return ' '.join(analysis_list)

def enhance_with_relevant_collocations(eng_explanation,mnc_sen_splitted):
    # if the entry has collocation information, add the relevant collocations to the basic entry
    if '\nCollocations:\n' in eng_explanation:
        eng_explanation_basic = eng_explanation.split('\nCollocations:\n')[0]
        collocations_pool = eng_explanation.split('\nCollocations:\n')[1].split('\n')
        # initialize the collocation as empty string, then iteratively add collocation if relevant
        collocations = ''
        for collocation in collocations_pool:
            entry,contecnt = collocation.split(":", 1)
            entry_roots = [re.sub('nco$','=',re.sub('mbi$','=',word)) for word in entry.strip().split(' ')]# handling both encrypted as unencrypted version
            #print(entry_roots)
            # if all roots in a collocation are in the morphologically analyzed sentence
            if all(root in mnc_sen_splitted for root in entry_roots):
                #print('relevant collocation found!')
                # add this collocation into the eng_basic
                collocations = collocations +'\n'+ entry +':'+ contecnt
                #print(collocations)
            else:
                collocations = collocations
        if collocations != '':# if some collocation has been found relevant, include it
            eng_explanation = eng_explanation_basic +'\nCollocations:'+ collocations
        else:
            eng_explanation = eng_explanation_basic
        return eng_explanation
    # if the entry does not have any collocation, return the entry as it is
    else:
        return eng_explanation

def get_manchu_entries(mnc_sen, collocations=True, suffixes=True, masked_out = False):
    entries_list = []
    # remove punctuation
    mnc_sen = re.sub(r'[^\w\s]', ' ', mnc_sen)
    mnc_sen = re.sub(r'\s+', ' ', mnc_sen)

    mnc_sen = re.sub('\(.*\)','',mnc_sen)
    mnc_sen = mnc_sen.strip()
    mnc_sen_splitted = morphology_analyzer_plus_assembled_dict(mnc_sen)
    #print(zho_sen,'\n',mnc_sen_splitted)
    # split suffix
    mnc_sen_splitted = re.sub(r'=', r'= =', mnc_sen_splitted)
    mnc_sen_splitted = re.sub(r'~', r' ~', mnc_sen_splitted)
    
    word_list = mnc_sen_splitted.split(' ')
    #print(word_list)
    for word in word_list:
        #print(word,convert2buleku_ortho(word))
        word = convert2buleku_ortho(word)
        if "/" not in word:
            eng_explanation = assembled_dict[re.sub('=$','mbi',word)]
            eng = enhance_with_relevant_collocations(eng_explanation,mnc_sen_splitted)
            entries_list.append(f"{word}: {eng}")
        # if the morhologically analyzed word has multiple possible analyses
        else:
            multiple_analyses = word.split('/')
            for m in multiple_analyses:
                #print(m)
                eng_explanation = assembled_dict[re.sub('=$','mbi',m)]
                eng = enhance_with_relevant_collocations(eng_explanation,mnc_sen_splitted)
                entries_list.append(f"{m}: {eng}")
    
    # remove collocations
    if collocations == False:
        for i in range(len(entries_list)):
            if 'Collocations' in entries_list[i]:
                collocations_removed = re.sub(r"\nCollocations:.*", "", entries_list[i],flags=re.DOTALL)
                entries_list[i] = collocations_removed
    # remove suffixes
    if suffixes == False:
        for entry in entries_list:
            if entry.startswith('~') or entry.startswith('='):
                del entries_list[entries_list.index(entry)]

    # masked out infrequent words, only keep the entry if the headword is in freq_token_list, 
    if masked_out == True:
        entries_list = [entry for entry in entries_list if re.match(r"^[^:]*", entry).group() in freq_token_list]
    return entries_list

def get_parallelSent(mnc_sen):
    parallelSent_set = set()
    # remove punctuation
    mnc_sen = re.sub(r'[^\w\s]', ' ', mnc_sen)
    mnc_sen = re.sub(r'\s+', ' ', mnc_sen)
    mnc_sen = re.sub('\(.*\)','',mnc_sen)
    mnc_sen = mnc_sen.strip()
    mnc_sen_splitted = convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_sen))
    #print(zho_sen,'\n',mnc_sen_splitted)
    # split suffix
    mnc_sen_splitted = re.sub(r'=', r'= =', mnc_sen_splitted)
    mnc_sen_splitted = re.sub(r'~', r' ~', mnc_sen_splitted)
    word_list = mnc_sen_splitted.split(' ')
    #print(word_list)
    for word in word_list:
        #print(word,convert2buleku_ortho(word))
        word = convert2buleku_ortho(word)
        if "/" not in word:
            parallel_sentences = word_to_sentence[re.sub('=$','mbi',word)]
            for parallel_sentence_pair in parallel_sentences:
                mnc_para, eng_para = parallel_sentence_pair.split('\n')
                # remove punctuation
                mnc_para = re.sub(r'[^\w\s]', ' ', mnc_para)
                mnc_para = re.sub(r'\s+', ' ', mnc_para)
                mnc_para = re.sub('\(.*\)','',mnc_para)
                mnc_para = mnc_para.strip()
                # make sure that the test sentence is not included in the parallel sentences
                if mnc_para != mnc_sen:
                    mnc_para = convert2buleku_ortho(morphology_analyzer_plus_assembled_dict(mnc_para))
                    parallelSent_set.add(f"Manchu sentence: {mnc_para}\nEnglish Translation: {eng_para}")
                # else:
                #     print('repetitive!')
        #     entries_list.append(f"{word}: {eng}")
        # # if the morhologically analyzed word has multiple possible analyses
        else:
            multiple_analyses = word.split('/')
            for m in multiple_analyses:
                parallel_sentences = word_to_sentence[re.sub('=$','mbi',m)]
                for parallel_sentence_pair in parallel_sentences:
                    mnc_para, eng_para = parallel_sentence_pair.split('\n')
                    # remove punctuation
                    mnc_para = re.sub(r'[^\w\s]', ' ', mnc_para)
                    mnc_para = re.sub(r'\s+', ' ', mnc_para)
                    mnc_para = re.sub('\(.*\)','',mnc_para)
                    mnc_para = mnc_para.strip()
                    # make sure that the test sentence is not included in the parallel sentences
                    if mnc_para != mnc_sen:
                        mnc_para = morphology_analyzer_plus_assembled_dict(mnc_para)
                        parallelSent_set.add(f"Manchu sentence: {mnc_para}\nEnglish Translation: {eng_para}")
                    # else:
                    #     print('repetitive!')
    return parallelSent_set

# retrieve the parallel examples using bm25
def get_parallelSent_bm25_top_n(analyzed_query,tokenized_corpus,mnc_eng_parallel_example_dict,n=10):
    tokenized_query = analyzed_query.replace(',',' ,').replace('.',' .').replace('=',' =').replace('~',' ~').split(' ')
    retreived_sens = bm25.get_top_n(tokenized_query, tokenized_corpus, n)

    retrieved_pairs = []
    for sen in retreived_sens:
        joined_sen = ' '.join(sen).replace(' ,',',').replace(' .','.').replace(' =','=').replace(' ~','~')
        # making sure that the retrieved sentence is not identical to the query
        if joined_sen != analyzed_query:
            retrieved_pairs.append(f"Manchu sentence: {joined_sen}\nEnglish Translation: {mnc_eng_parallel_example_dict[joined_sen]}")
        # else:
        #     print('same!',joined_sen)
    return retrieved_pairs

# get the grammatical suffix set from an input sentence
def get_suffix_set(mnc_sen):
    suffix_set = set()
    # remove punctuation
    mnc_sen = re.sub(r'[^\w\s]', ' ', mnc_sen)
    mnc_sen = re.sub(r'\s+', ' ', mnc_sen)

    mnc_sen = re.sub('\(.*\)','',mnc_sen)
    mnc_sen = mnc_sen.strip()
    mnc_sen_splitted = morphology_analyzer_plus_assembled_dict(mnc_sen)
    #print(zho_sen,'\n',mnc_sen_splitted)
    # split suffix
    mnc_sen_splitted = re.sub(r'=', r'= =', mnc_sen_splitted)
    mnc_sen_splitted = re.sub(r'~', r' ~', mnc_sen_splitted)
    
    word_list = mnc_sen_splitted.split(' ')
    #print(word_list)
    for word in word_list:
        #print(word,convert2buleku_ortho(word))
        word = convert2buleku_ortho(word)
        if word in suffix_to_gloss_dict.keys():
            suffix_set.add(word)
        if "/" not in word:
            if word in suffix_to_gloss_dict.keys():
                suffix_set.add(word)
        # if the morhologically analyzed word has multiple possible analyses
        else:
            multiple_analyses = word.split('/')
            for m in multiple_analyses:
                #print(m)
                if m in suffix_to_gloss_dict.keys():
                    suffix_set.add(m)
    return suffix_set

# get the grammar sections from a grammar_dict (long or short), for a given sentence
def get_grammar_sections(sen,grammar_dict):
    grammar_sections = 'Below are detailed grammatical explanations for the suffixes used in the given sentences:\n'
    suffix_set = get_suffix_set(sen)
    #print(suffix_set)
    for suffix in suffix_set:
        gloss = suffix_to_gloss_dict[suffix]
        grammar_sections += f"{suffix}: {gloss}\n"
        grammar_sections += grammar_dict[gloss]
    return grammar_sections

# for encrypting manchu
def next_consonant(char):
    consonants = "bcdfghjklmnpqrstvwxyz"
    if char in consonants:
        index = consonants.index(char)
        return consonants[(index + 1) % len(consonants)]
    return char

def next_vowel(char):
    vowels = "aeiou"
    if char in vowels:
        index = vowels.index(char)
        return vowels[(index + 1) % len(vowels)]
    return char

def transform_word(word):
    result = []
    for char in word.lower():
        if char in "bcdfghjklmnpqrstvwxyz":
            result.append(next_consonant(char))
        elif char in "aeiou":
            result.append(next_vowel(char))
        else:
            result.append(char)
    return ''.join(result)

def transform_sen(original_sen):
    return ' '.join([transform_word(w) for w in original_sen.split(' ')])

def get_manchu_entries_encrypted(mnc_sen, collocations=True, suffixes=True, masked_out=False):
    entries_list = []
    # remove punctuation
    mnc_sen = re.sub(r'[^\w\s]', ' ', mnc_sen)
    mnc_sen = re.sub(r'\s+', ' ', mnc_sen)

    mnc_sen = re.sub('\(.*\)','',mnc_sen)
    mnc_sen = mnc_sen.strip()

    mnc_sen_splitted = morphology_analyzer_plus_assembled_dict(mnc_sen)
    #print(zho_sen,'\n',mnc_sen_splitted)
    # split suffix
    mnc_sen_splitted = re.sub(r'=', r'= =', mnc_sen_splitted)
    mnc_sen_splitted = re.sub(r'~', r' ~', mnc_sen_splitted)
    
    word_list = mnc_sen_splitted.split(' ')
    word_list = [convert2buleku_ortho(word) for word in word_list]
    #print(word_list)

    # encrypted the words
    encrypted_word_list = [transform_word(w) for w in word_list]

    for word in encrypted_word_list:
        if "/" not in word:
            eng_explanation = assembled_dict_encrypted[re.sub('=$','nco',word)]
            eng = enhance_with_relevant_collocations(eng_explanation,' '.join(encrypted_word_list))
            entries_list.append(f"{word}: {eng}")
        # if the morhologically analyzed word has multiple possible analyses
        else:
            multiple_analyses = word.split('/')
            for m in multiple_analyses:
                #print(m)
                eng_explanation = assembled_dict_encrypted[re.sub('=$','nco',m)]
                eng = enhance_with_relevant_collocations(eng_explanation,' '.join(encrypted_word_list))
                entries_list.append(f"{m}: {eng}")
    
    # remove collocations
    if collocations == False:
        for i in range(len(entries_list)):
            if 'Collocations' in entries_list[i]:
                collocations_removed = re.sub(r"\nCollocations:.*", "", entries_list[i],flags=re.DOTALL)
                entries_list[i] = collocations_removed
    # remove suffixes
    if suffixes == False:
        for entry in entries_list:
            if entry.startswith('~') or entry.startswith('='):
                del entries_list[entries_list.index(entry)]

    # masked out infrequent words, only keep the entry if the headword is in freq_token_list, 
    if masked_out == True:
        entries_list = [entry for entry in entries_list if re.match(r"^[^:]*", entry).group() in freq_token_list]
    return entries_list

def encrypt_parallelSent(parallelSents):
    encrpted_parallelSent_set = set()
    for pair in parallelSents:
        mnc_sen, eng_sen = pair.split('\n')
        mnc_sen_encrypted = mnc_sen.split(':')[0].replace('Manchu','Unknown language') + ':' + transform_sen(mnc_sen.split(':')[1])
        encrpted_parallelSent_set.add(f"{mnc_sen_encrypted}\n{eng_sen}")
    return encrpted_parallelSent_set