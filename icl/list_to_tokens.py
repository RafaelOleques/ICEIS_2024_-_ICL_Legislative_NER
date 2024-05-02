def extract_entity_type(self, words, terms_distance_sorted, entities):
    scores = [None] * len(entities)

    for key in terms_distance_sorted:
        splited = terms_distance_sorted[key]["term"].split()
        indices = self.find_sequence_in_list(words, splited)

        for indice in indices:
            subcategories = self.get_label(terms_distance_sorted[key]["thesaurus_term"])
            score = terms_distance_sorted[key]["distance"]
            subcategories = self.disambiguation_category(subcategories)

            i, j = indice
            j = j+1

            entities[i:j] = [f"B-{subcategories}" if k == i else f"I-{subcategories}" for k in range(i, j)]
            scores[i:j] = [score for k in range(i, j)]
            
    return entities, scores

def find_sequence_in_list(self, lst, sequence):
    indices = []
    sequence_length = len(sequence)

    for i in range(len(lst) - sequence_length + 1):
        if lst[i:i+sequence_length] == sequence:
            indices.append((i, i + sequence_length - 1))

    return indices