import os
import evaluate
import datasets
from docopt import docopt
from transformers import pipeline

usage= """
    Usage:  evaluate.py --model <model_name> --dataset <dataset_path>
"""

def argParser(arg):
    if not arg['<model_path>']:
        raise Exception("Introduzca el nombre del modelo")
    if not arg['<dataset_path>']:
        raise Exception("Introduzca la ruta del conjunto de datos")
    return arg['<model_path>'], arg['<dataset_path>']

def calculate_bleu(predictions, references):
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    print(results['bleu'])
    
def get_test_answers(dataset):
    return dataset['test']['answer'].tolist()  

def get_question_and_contexts(x):
    return {'question': x['question'], 'context': x['context']}

def main():
    arg = docopt(usage)
    model_name, dataset_path = argParser(arg)
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    dataset = datasets.load_from_disk(dataset_path)
    
    qa = list(map(get_question_and_contexts, dataset['test'])) 
    predictions = nlp(qa)
    references = get_test_answers(dataset)
    calculate_bleu(predictions, references)
    
if __name__ == "__main__":
    main()