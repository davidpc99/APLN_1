from docopt import docopt
from transformers import pipeline
from alive_progress import alive_bar
import datasets
import pysolr
import torch

usage= """
    Usage:  solr.py send
        solr.py retrieve [--query] <query>
"""

def send():
    solr = pysolr.Solr('http://localhost:8983/solr/prac1', always_commit=True)

    dataset = datasets.load_from_disk("data/wiki_corpus")
    dataset = dataset.select(range(10000))
    documents = [{"id": row['id'], "title": row['title'], "description": row['text']} for row in dataset]
    solr.add(documents)
    

def retrieve(query):
    solr = pysolr.Solr('http://localhost:8983/solr/prac1', always_commit=True)

    documents = solr.search(query, rows=10, fl=["description"])
    return documents


def send_to_pipeline(documents, query):
    device_id = -1
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    model_name = "Rakib/roberta-base-on-cuad"
    batch = [{'question': query, 'context': document['description']} for document in documents]

    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device_id, batch_size=len(documents))
    print("Getting predictions...")
    predictions = nlp(batch)
    score_list = [prediction['score'] for prediction in predictions]
    best_prediction_index = score_list.index(max(score_list))
    print(predictions[best_prediction_index]['answer'])


def argParser(arg):
    if arg['send']:
        return send, None
    elif arg['retrieve']:
        query = arg['<query>']
        return retrieve, query
    else:
        raise Exception("Introduzca los argumentos correctamente")


def main():
    arg = docopt(usage)
    selected_function, query = argParser(arg)

    if selected_function is send:
        selected_function()
    else:
        documents = selected_function(query)
        send_to_pipeline(documents, query)



if __name__ == "__main__":
    main()