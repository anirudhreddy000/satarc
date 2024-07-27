from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

def load_data(file_path):
    examples = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            parts = line.split('\t')
            if len(parts) >= 3:
                sentence1 = parts[0]
                sentence2 = ' '.join(parts[1:-1])  
                score = parts[-1]
                try:
                    score = float(score)
                    examples.append(InputExample(texts=[sentence1, sentence2], label=score))
                except ValueError:
                    print(f"Skipping line due to conversion error: {line}")
            else:
                print(f"Skipping line due to format error: {line}")
    return examples


file_path = 'train_data_2.txt'
nasa_examples = load_data(file_path)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

train_dataset = SentencesDataset(nasa_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

train_loss = losses.CosineSimilarityLoss(model)

model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(nasa_examples, name='sts-eval')
model.evaluate(evaluator)

model.save('fine-tuned-model-3')
