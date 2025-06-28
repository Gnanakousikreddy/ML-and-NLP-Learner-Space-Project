# importing the required libraries
from datasets import load_dataset, DatasetDict
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
from transformers import TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import f1_score
import torch


# loading the data
print("Loading the data...")
imdb_dataset = load_dataset("imdb")
imdb = DatasetDict(train = imdb_dataset['train'].shuffle(seed = 1111).select(range(20000)),
                    val = imdb_dataset['train'].shuffle(seed = 1111).select(range(20000, 25000)),
                    test = imdb_dataset['test']
                    )

print("Dataset :")
print(imdb)


# tokenizing the data
print("Tokenizing the data...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenized_dataset = imdb.map(
    lambda example : tokenizer(example['text'], padding = "max_length", max_length= 256, truncation=True),
    batched = True,
    batch_size = 64,
)

tokenized_dataset = tokenized_dataset.remove_columns(['text'])
tokenized_dataset = tokenized_dataset.rename_columns({'label':'labels'})
tokenized_dataset.set_format("torch")

print("Tokenized Dataset...")
print(tokenized_dataset)


# Fine tuning the model using trainer api
print("Fine tuning the model...")

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1
    }

arguments = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    fp16=True,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224,
    report_to="none"
)


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["val"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("Training...")
trainer.train()


# Saving the best model
print("Saving the Model...")
trainer.model.save_pretrained("./best_model")
trainer.tokenizer.save_pretrained("./best_model")

# Results
results = trainer.evaluate(tokenized_dataset['test'])
print("Results :")
print(results)

# Loading the model
print("Loading the saved model...")
loaded_model = BertForSequenceClassification.from_pretrained("./best_model")
loaded_tokenizer = BertTokenizer.from_pretrained("./best_model")


# predicting the sentiment for a given sentence 
def predict_sentiment(model, tokenizer, text) :
  model_inputs = tokenizer(text, return_tensors='pt')
  pred = torch.argmax(loaded_model(**model_inputs).logits)
  return ['NEGATIVE', 'POSITIVE'][pred]

print("Predicting the sentence for the text :")
text = "the movie was not good"
print(text)
print("SENTIMENT :", predict_sentiment(loaded_model, loaded_tokenizer, text))

print("Completed...")