from datasets import load_dataset
import pandas as pd
import os

def save_as_csv(dataset, filename):
    df = pd.DataFrame(dataset)
    df.to_csv(filename, index=False)


def main():
    dir = "data"
    # Si el fichero no existe guardamos el dataset
    
    if not os.path.isdir(dir):
        os.mkdir(dir)
        #dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]+train[20%:21%]+train[40%:41%]+train[60%:61%]+train[80%:81%]")
        dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
        dataset.save_to_disk("data/wiki_corpus")
        dataset = load_dataset("wiki_qa")
        dataset.save_to_disk("data/qa_corpus")

if __name__ == "__main__":
    main()