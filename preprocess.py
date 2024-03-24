import datasets
from alive_progress import alive_bar


def main():    
    dataset1 = datasets.load_from_disk("data/wiki_corpus")
    dataset2 = datasets.load_from_disk("data/qa_corpus")
    
    print(dataset1[0]['title'])
    print(dataset1[6407813]['title'])
    
    title_to_context = {}
    #{example['title']: example['text'] for example in dataset1}
    with alive_bar(len(dataset1), title="Title to Content Mapping") as bar:
        for example in dataset1:
            title_to_context[example['title']] = example['text']
            bar()
    
    dataset2_mapped = {
        key: {**value, "context": title_to_context.get(value['title'])}
        for key, value in dataset2.items()
    }
    dataset2_mapped_datadict = datasets.DatasetDict(dataset2_mapped)

    print(dataset2_mapped_datadict['test'][0])
    

if __name__ == "__main__":
    main()