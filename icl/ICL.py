import time
import maritalk

import pandas as pd
from .sampler import Sampler
from .configs import *

import os

class ICL:

    def __init__(self, file_path, labels_list, type_file=TXT) -> None:

        if type_file == TXT:
            self.labeled_corpus = self.create_dataframe_from_txt(file_path)
        else:
            self.labeled_corpus = pd.read_json(file_path)

        self.labeled_corpus = self.labeled_corpus.reset_index(drop=True)
        self.model = maritalk.MariTalk(key=os.environ["key"])
        self.sampler = Sampler("distiluse-base-multilingual-cased", SBERT, self.labeled_corpus, "sentences", labels_list)

    def create_dataframe_from_txt(self, file_path):
        # Read the text file and split each line into token and NER class
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        data = []
        for line in lines:
            if line != '' and line != '\n':
                line = line.strip().split(' ')
                token = line[0].lower()
                ner_token = line[1]
                data.append((token, ner_token))
            else:
                token = '\n'
                ner_token = '\n'
                data.append((token, ner_token))

        # Group tokens into sentences considering the '\n' line breaks
        sentences = []
        tokens = []
        ner_tokens = []
        sentence = []
        ner_token = []

        for token, ner_token_str in data:
            if token != '\n':
                sentence.append(token)
                ner_token.append(ner_token_str)
            else:
                sentences.append(' '.join(sentence))
                tokens.append(sentence)
                ner_tokens.append(ner_token)
                sentence = []
                ner_token = []

        # Create the DataFrame
        df = pd.DataFrame({
            'sentences': sentences,
            'tokens': tokens,
            'ner_tags': ner_tokens
        })
        
        df['num_tokens'] = df['sentences'].str.split().apply(len)
        df = df[df['num_tokens'] > 1].drop(columns='num_tokens')
                                        
        return df


    def select_example(self, index):
        sentence = self.labeled_corpus.iloc[index].sentences
        entities = []
        
        same = False
        term = ""
        entity = ""
        
        for token, ner_token in zip(self.labeled_corpus.iloc[index].tokens, self.labeled_corpus.iloc[index].ner_tags):
            if ner_token == "O":
                if term != "":
                    entities.append(f"{term} eh um {entity}")
                #entities.append(f"({token}, {ner_token})")
                term = ""
                entity = ""
                same = False
            elif same == False:
                term = token
                entity = ner_token.replace("B-", "")
                entity = entity.replace("I-", "")
                
                same = True
            else:
                follow_entity = ner_token.replace("B-", "")
                follow_entity = follow_entity.replace("I-", "")
                
                if entity == follow_entity and "I-" in ner_token:
                    term += " "+ token
                else:
                    entities.append(f"{term} eh um {entity}")
                    term = ""
                    entity = ""
                    same = False
            
            
        
        #entities = [f"({token}, {ner_token})" for token, ner_token in zip(self.labeled_corpus.iloc[index].tokens, self.labeled_corpus.iloc[index].ner_tags)]
        
        example = f"""Sentença: {sentence}
        
Termos com categorias: {"; ".join(entities)}
    """
        
        return example

    def generate_examples(self, sentence, metric, seed, reverse, k):
        examples = ""

        if metric == K_SIMILAR:
            ids = self.sampler.k_similar(query=sentence, k=k, reverse=reverse)
        elif metric == K_SIMILAR_PER_CATEGORIES:
            ids = self.sampler.k_similar_per_categories(query=sentence, k=k, reverse=reverse)
        elif metric == RANDOM:
            ids = self.sampler.random(k=5, seed=seed)

        for i in ids:
            examples += self.select_example(i)+"\n"

        return examples

    def prompt(self, command, sentence, categories, metric, k, seed=SEED, reverse=False):
        prompt = f"""{command}

{categories}

{self.generate_examples(sentence, metric, seed, reverse, k)}
Sentença: {sentence}
Termos com categorias:"""
    
        while(True):
            try:
                answer = self.model.generate(
                    prompt,
                    chat_mode=False,
                    do_sample=False,
                    max_tokens=1000,
                    stopping_tokens=["\n"]
                )

                break
            except:
                time.sleep(5)
       
        return prompt, answer
        
    def convert_to_bio(self, answer):
        entities = [result.split(" eh um ") for result in answer.split("; ")]
        
        ner_entities = []

        for tuple in entities:
            if len(tuple) != 2:
                continue
            else:
                sentence, entity = tuple

                ner_entities.append([])

                tokens = sentence.split(" ")

                ner_entities[-1].append((tokens[0], f"B-{entity}"))

                for i in range(1, len(tokens)):
                    ner_entities[-1].append((tokens[i], f"I-{entity}"))
                
        return ner_entities
