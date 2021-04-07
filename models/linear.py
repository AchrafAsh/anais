import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, embed_dim, src_vocab_size, target_vocab_size):
        super(LinearModel, self).__init__()
        self.embed_dim = embed_dim
        self.src_vocab_size = src_vocab_size
        self.target_vocab_size = target_vocab_size

        self.embeddings = nn.Linear(in_features=src_vocab_size, out_features=embed_dim)
        self.out = nn.Linear(in_features=embed_dim, out_features=target_vocab_size)

    def forward(self, x):
        # x (batch_size, src_vocab_size, word_len)
        embeddings = self.embeddings(x.permute(0, 2, 1)) # (batch_size, word_len, embed_dim)
        output = self.out(embeddings) # (batch_size, word_len, target_vocab_size)
        output = output.sum(dim=1) # (batch_size, target_vocab_size)
        return F.softmax(output, dim=1) # (batch_size, target_vocab_size)


def train(model, dataset, lr, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.params(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0

        for _, batch in enumerate(dataset):
            optimizer.zero_grad()
            
            sentences, targets = zip(batch)
            
            preds = model(sentences)
            loss = criterion(preds, targets)

            total_loss += loss
            
            optimizer.step()

        print(f"epoch: [{epoch / epochs}] | loss: [{total_loss:.2}]")