import pandas as pd
from tqdm import tqdm

from icl.configs import *
from icl.ICL import ICL

#labels = pd.read_json('./corpora/lener/labels.json')
labels = pd.read_json('./corpora/ulysses/labels.json')

labels_list = [label.replace("B-", "") for label in labels.labels.values.tolist() if "I-" not in label and "O" != label]

#icl = ICL('./corpora/lener/train.txt', labels_list, type_file=TXT)
icl = ICL('./corpora/ulysses/train.txt', labels_list, type_file=TXT)

#df_teste = pd.read_json('./corpora/lener/test.json')
df_teste = icl.create_dataframe_from_txt('./corpora/ulysses/test.txt')

corpora_teste = df_teste["sentences"].values.tolist()

command = "Reconheça os termos significativos e suas categorias."

categories = f"As categorias possíveis são: {', '.join(labels_list)}"

k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

for k in k_list:
    metrics = [K_SIMILAR_PER_CATEGORIES, K_SIMILAR, RANDOM]
    reverse_list = [True, False]

    for metric in metrics:
        for reverse in reverse_list:
            data_results = {"prompts": [], "sentences": [], "results": []}
            
            path = f"./corpora/icl/{metric}"
            if reverse:
                path += "_reverse"

            print(path)

            for sentence in tqdm(corpora_teste):
                prompt, answer = icl.prompt(command=command, sentence=sentence, categories=categories, metric=metric, k=k, reverse=reverse)
                result = icl.convert_to_bio(answer)

                data_results["sentences"].append(sentence)
                data_results["results"].append(result)
                data_results["prompts"].append(prompt)

            df = pd.DataFrame(data_results)

            df.to_json(f"{path}_{k}.json", orient = 'records', compression = 'infer', default_handler=str)