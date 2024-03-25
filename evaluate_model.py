from docopt import docopt
from transformers import pipeline
from alive_progress import alive_bar
import torch
import datasets
import evaluate

usage= """
    Usage:  evaluate_model.py [--model] <model_name> [--dataset] <dataset_path>
"""

def argParser(arg):
    if not arg['<model_name>']:
        raise Exception("Introduzca el nombre del modelo")
    if not arg['<dataset_path>']:
        raise Exception("Introduzca la ruta del conjunto de datos")
    return arg['<model_name>'], arg['<dataset_path>']

def calculate_bleu(predictions, references):
    references = [[reference] for reference in references]
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=predictions, references=references)
    return results

def calculate_rouge(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results
    
def get_test_answers(dataset):
    #return [[string] for string in dataset['test']['answer']]
    return dataset['test']['answer']

def get_predicted_answers(predictions):
    return [prediction["answer"] for prediction in predictions]
    ##return predictions["answer"]

def get_question_and_contexts(x):
    return {'question': x['question'], 'context': x['context']}


def main():
    device_id = -1
    if torch.cuda.is_available():
        device_id = torch.cuda.current_device()

    arg = docopt(usage)
    model_name, dataset_path = argParser(arg)
    dataset = datasets.load_from_disk(dataset_path)
    qa = list(map(get_question_and_contexts, dataset['test']))
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, device=device_id, batch_size=len(qa))

    print("Getting predictions...")
    predictions = get_predicted_answers(nlp(qa))
    references = get_test_answers(dataset)


    bleu_score = calculate_bleu(predictions, references)
    rouge_score = calculate_rouge(predictions, references)
    scores = {"bleu": bleu_score["bleu"], "rouge1": rouge_score["rouge1"], "rouge2": rouge_score["rouge2"], "rougeL": rouge_score["rougeL"]}
    print(scores)
    
if __name__ == "__main__":
    main()