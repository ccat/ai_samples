from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm


def main():
    """BERTモデルの読み込みから学習まで一通り実施して、学習結果のモデルを返します。
    """
    print("start")
    model,tokenizer = make_model() #はじめに学習対象となるモデルを作成します
    print("model loaded")
    dataset = load_and_tokenize_datasets(tokenizer) #学習用のデータを読み込みます
    print("dataset loaded")
    train(model,dataset, 1) #学習を行います
    print("trained")
    return model


def make_model():
    """BERTのモデルを読み込み、トークナイザーを生成してモデルとトークナイザーをセットで返します。
    """
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return (model,tokenizer)


def load_and_tokenize_datasets(tokenizer):
    """GLUE BenchmarkのMRPCのデータを読み込み、学習に投入できるデータ構造に変換してデータセットを返します
    """
    def encode(examples):
        return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, padding='max_length')

    dataset = load_dataset('glue', 'mrpc', split='train')
    dataset = dataset.map(encode, batched=True)
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])
    return dataset


def train(model,dataset,epoch_range=3):
    """変換済みデータセットでモデルの学習を実行します
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.train().to(device)
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
    for epoch in range(epoch_range):
        for i, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if i % 10 == 0:
                print(f"loss: {loss}")

if __name__ == '__main__':
    main()
