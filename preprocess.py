import datasets
from alive_progress import alive_bar
import pandas

def remove_duplicates(dataset):
    for set in dataset:
        dataset[set] = datasets.Dataset.from_pandas(pandas.DataFrame(dataset[set]).drop_duplicates())
    return dataset



def main():    
    dataset1 = datasets.load_from_disk("data/wiki_corpus")
    dataset2 = datasets.load_from_disk("data/qa_corpus")

    # Eliminar duplicados
    dataset2 = remove_duplicates(dataset2)

    #dataset1 = datasets.Dataset.from_dict(dataset1[0:100000])

    # Elinminar etiquetas con ceros (malas respuestas)
    dataset2 = dataset2.filter(lambda x: x["label"]==1)

    for key, _ in dataset2.items():
        dataset2[key] = dataset2[key].add_column("context", [""]*len(dataset2[key]))

    title_set = set()
    with alive_bar(len(dataset2["train"])+len(dataset2["validation"])+len(dataset2["test"]), title="Title to Content Mapping") as bar:
        for _, dataset_set in dataset2.items():
            for example in dataset_set:
                title_set.add(example['document_title'])
                bar()

    title_to_context = {}
    with alive_bar(len(dataset1), title="Title to Content Mapping") as bar:
        for example in dataset1:
            if example['title'] in title_set:
                title_to_context[example['title']] = example['text']
            bar()

    for dataset_set in dataset2:
        dataset2[dataset_set] = dataset2[dataset_set].map(lambda x: {'context': title_to_context.get(x['document_title'], "")})
    
    dataset2 = dataset2.filter(lambda x: x["context"]!="")

    print(dataset2)
    dataset2.save_to_disk("data/preprocessed_qa")

if __name__ == "__main__":
    main()