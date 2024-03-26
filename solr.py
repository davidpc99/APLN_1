import pysolr
from docopt import docopt
import datasets
import os

usage= """
    Usage:  solr.py send
        solr.py retrieve [--query] <query>
"""

def send():
    solr = pysolr.Solr('http://localhost:8983/solr/prac1', always_commit=True)

    dataset = datasets.load_from_disk("data/wiki_corpus")
    dataset = dataset.select(range(1000))
    documents = [{"id": row['id'], "title": row['title'], "description": row['text']} for row in dataset]
    solr.add(documents)
    

def retrieve(query):
    solr = pysolr.Solr('http://localhost:8983/solr/prac1', always_commit=True)

    documents = solr.search(query, rows=2, fl=["description"])
    if not os.path.exists("solr_retrieved"):
        os.mkdir("solr_retrieved")

    f = open("solr_retrieved/solr_documents.txt", "w", encoding="utf-8")
    for document in documents:
        f.write(str(document))
    f.close()


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
        selected_function(query)

if __name__ == "__main__":
    main()