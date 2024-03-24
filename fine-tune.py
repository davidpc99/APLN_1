from transformers import AutoModelForQuestionAnswering, AutoTokenizer




def main():
    model_name = "deepset/roberta-base-squad2"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

if __name__ == "__main__":
    main()