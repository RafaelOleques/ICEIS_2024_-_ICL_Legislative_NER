from sentence_transformers import SentenceTransformer, util
import random
from operator import itemgetter

from .configs import SBERT

class Sampler:
    def __init__(self, model_name: str, embedding_model_type: str, df: list, input, labels_list: list) -> None:
        self.model = self.__set_embedding_model(model_name, embedding_model_type)
        self.df = df
        self.corpus = self.df[input].tolist()
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        device = self.corpus_embeddings.device

        self.categories = labels_list
        
        self.categories_embeddings = {}
        self.categories_indexes = {}

        for categorie in self.categories:
            df_categorie = self.df[self.df['ner_tags'].apply(lambda x: any(categorie in token for token in x))]
            self.categories_embeddings[categorie] = self.model.encode(df_categorie[input].tolist(), convert_to_tensor=True)
            self.categories_indexes[categorie] = {idx: val for idx, val in enumerate(df_categorie.index.tolist())}

    def __set_embedding_model(self, model_name: str, embedding_model_type: str):
        if embedding_model_type == SBERT:
            model = SentenceTransformer(model_name)
        else:
            print("Wrong model type")
            model = None

        return model
    
    def k_similar(self, query, k: int,  threshold: float=0, reverse: bool = False) -> list:       
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        #Selecting
        search_hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=k)
        search_hits = [hit for hit in search_hits[0] if hit["score"] > threshold]

        ##Sentences
        ids = [hit["corpus_id"] for hit in search_hits]

        if reverse:
            ids = ids[::-1]
        
        return ids
        
    def k_similar_per_categories(self, query, k: int,  reverse: bool = False) -> list:       
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        #Selecting
        ids = {}

        for categorie in self.categories:
            search_hits = util.semantic_search(query_embedding, self.categories_embeddings[categorie], top_k=1)[0]

            if self.categories_indexes[categorie][search_hits[0]["corpus_id"]] not in ids:
                ids[self.categories_indexes[categorie][search_hits[0]["corpus_id"]]] = search_hits[0]["score"]

        if len(ids) < k:
            search_hits = util.semantic_search(query_embedding, self.corpus_embeddings)[0]

            for hit in search_hits:
                if hit["corpus_id"] not in ids:
                    ids[hit["corpus_id"]] = search_hits[0]["score"]

                if len(ids) < k:
                    break

        ids =  dict(sorted(ids.items(), key=itemgetter(1), reverse=True))
        ids_list = list(ids.keys())

        if reverse:
            ids_list = ids_list[::-1]
            
        return ids_list
    
    def random(self, k: int, seed:int):
        random.seed(seed)

        return random.sample(range(len(self.corpus)), k)


        