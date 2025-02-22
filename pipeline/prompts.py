def extract_enclosed_text(text, boundary="###"):
    splits = text.split(boundary)
    if len(splits) < 3:
        return ""
    return splits[-2].strip()

def prompt_system(lang):
    return "You are a linguistic expert who never refuses to use your knowledge to help others."

def prompt_direct_translate(src_lang, tgt_lang, sent):
    prompt = f"""Please help me translate the following sentence from {src_lang} to {tgt_lang}:
{sent}
Please try your best to translate, it's okay if your translation is bad. Do not refuse to try it. I won't blame you.
Please enclose your translation in ###.
For example, if your translation is "Hello world", the last part of your output should be ### Hello world ###.
"""
    return prompt

def prompt_direct_translate_morph(src_lang, tgt_lang, segmented_sent):
    prompt = f"""Please help me translate the following sentence from {src_lang} to {tgt_lang}:
{segmented_sent}
The morphemes in this sentence have been segmented: the verb stem and verbal suffixes are separated by '=', the noun stem and nominal suffixes are separated by '~'. 
Please try your best to translate, it's okay if your translation is bad. Do not refuse to try it. I won't blame you.
Please enclose your translation in ###.
For example, if your translation is "Hello world", the last part of your output should be ### Hello world ###.
"""
    return prompt

def prompt_template(src_lang, tgt_lang, sent, wordbyword, components_list):
    components = '\n'.join(components_list)
    prompt = f"""
Please help me translate the following sentence from {src_lang} to {tgt_lang}:
{sent}
The morphemes in this sentence have been segmented: the verb stem and verbal suffixes are separated by '=', the noun stem and nominal suffixes are separated by '~'. 
Note that some words can be either analyzed as a whole (e.g. 'oho'), or as a word stem plus a suffix (e.g. o=ho), the different analyses are separated by '/'. In such case, explanations for both analyses are given below, and you need to choose which one is the most appropriate in the given context.

For the translation task, you are given the word by word mapping from the {src_lang} words to the {tgt_lang} words.
Some words can be polysemous and there might be multiple possible English translations. In such case, please choose the most appropriate one.
Note that for some words, they might be derived from a more basic form, we call this the parent word. The parents are also given in the word by word translation.
Here are the dictionary entries for each individual word in the source sentence:
{wordbyword}

Note that sometimes two or more words can form a collocation and express a specific meaning. You should refer to the collocations listed under the dictionary entries. 
For example, 'mama' means 'grandmother', 'erxe=' means 'to attend', but 'mama erxe=' as a collocation means 'to get smallpox'.
In such case, explain which collocation meaning you think is most appropriate in the context.

{components}

Using all the information provided above, now please translate the sentence into {tgt_lang}.
Remember your source sentence is: {sent}
Please make sure to enclose your final translation in ###. For example, if your translation is "Hello world", the last part of your output should be ###Hello world###.
"""
    return prompt

def component_cot_Ca(src_lang,tgt_lang):
    return f"""Given the previous information, please first annotate the meaning and grammatical features of each word in the sentence.
For each word, based on their English transaltion and whether it ends with '='(marker of verb stems), firstly decide whether the word is nominal (noun/adjective), or a verbal(verb, converb) or else (other part of speech such as adverb, postposition ect.).
Then for each noun, please annotate its number (singular/plural) and case (Nominative/Genitive/Dative-Locative/Accusative/Ablative), based on the particles/suffixes that follow the noun.
And for each verb, please annotate its tense (perfect/imperfect) and form (Affirmative/Negative/Interrogative/Imperative/Optative/Desiderative), based on the suffixes attached to the verb.

Then based on the annotations, translate the sentence from {src_lang} into {tgt_lang} based on the annotations and the analyzed sentence structure. """

def component_cot_Cas(src_lang,tgt_lang):
    return f"""Given the previous information, please proceed with the following steps:
Step 1: 
Please first annotate the meaning and grammatical features of each word in the sentence.
For each word, based on their English transaltion and whether it ends with '='(marker of verb stems), firstly decide whether the word is nominal (noun/adjective), or a verbal(verb, converb) or else (other part of speech such as adverb, postposition ect.).
Then for each noun, please annotate its number (singular/plural) and case (Nominative/Genitive/Dative-Locative/Accusative/Ablative), based on the particles/suffixes that follow the noun.
And for each verb, please annotate its tense (perfect/imperfect) and form (Affirmative/Negative/Interrogative/Imperative/Optative/Desiderative), based on the suffixes attached to the verb.
Step 2:
Then based on the annotations, please analyze the sentence structure by figuring out what the subject and object of each verb is. Keep in mind that {src_lang}'s basic word order is subject–object–verb (SOV) and it is a head-final language, so that the adjectives and participles always precede the noun they modifies, and the arguments to the verb always precede the verb.
Note that clauses can be combined into a single sentence by using converbs, which relate the first action to the second.
The final step:
Translate the sentence into {tgt_lang} based on the annotations and the analyzed sentence structure. """

def component_para(src_lang,tgt_lang,parallel_sent):
    return f"""To help with the translation, here are some {src_lang}-{tgt_lang} parallel sentences that may be helpful for your translation:
{parallel_sent}"""

def component_grammar(grammar_sections):
    return f"""
You are also given this grammar book below. Feel free to rely on this grammar book in your translation task:

- Manchu Grammar Book
The Manchu language is typologically similar to the Mongolic and Turkic languages. 
All Manchu phrases are head-final; the head-word of a phrase (e.g. the noun of a noun phrase, or the verb of a verb phrase) always falls at the end of the phrase. 
Thus, adjectives and adjectival phrases always precede the noun they modify, and the arguments to the verb always precede the verb. 
As a result, Manchu sentence structure is subject–object–verb (SOV).
Manchu also makes extensive use of converb structures and has an inventory of converbial suffixes to indicate the relationship between the subordinate verb and the finite verb that follows it.
Unlike English, which uses prepositions, Manchu exclusively uses postpositions.
The Manchu language is agglutinative in word structure, meaning that words are formed by adding suffixes to the root, and each morpheme in a word has one distinct meaning or grammatical function.

{grammar_sections}"""