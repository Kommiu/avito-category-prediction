import torch
import torch.nn as nn
import torch.nn.functional as F
from training_utils import categorical_accuracy

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers,
                 biderectional, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           biderectional=biderectional,
                           dropout=dropout
                           )

        self.fc = nn.Linear(hidden_dim*2, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, text_lengths):

        embedded = self.dropout(self.embedding(text))

        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)

        _, (hidden, cell) = self.rnn(packed_embedded)

        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)

        return self.fc(hidden.squeeze(0))



class CNN(nn.Module):

    def __init__(self, vocab_size, output_dim, embedding_dim, n_filters, filter_sizes,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]

        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def train_epoch(self, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.train()

        for batch in iterator:
            optimizer.zero_grad()
            predictions = self.forward(batch.text)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss/len(iterator), epoch_acc/len(iterator)

    def  evaluate_epoch(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        self.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = self.forward(batch.text)

                loss = criterion(predictions, batch.label)
                acc = categorical_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss/len(iterator), epoch_acc/len(iterator)

class CNN_jaccard(nn.Module):
    def __init__(self, vocab_size, output_dim, embedding_dim,
                 n_filters, filter_sizes, linear_sizes,
                 dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1,
                      out_channels=n_filters,
                      kernel_size=(fs, embedding_dim))
            for fs in filter_sizes
        ])
        self.mlp = nn.Sequential()
        for i, (in_dim, out_dim) in enumerate(zip(linear_sizes[:-1], linear_sizes[1:])):
            self.mlp.add_module(f'l-{i}', nn.Linear(in_dim, out_dim))
            self.mlp.add_module(f'a-{i}', nn.ReLU())

        self.fc = nn.Linear(len(filter_sizes) * n_filters + linear_sizes[-1], output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, text, nums):
        # text = [sent len, batch size]

        text = text.permute(1, 0)
        # text = [batch size, sent len]
        nums = nums.permute(1, 0)
        embedded = self.embedding(text)
        # embedded = [batch size, sent len, emb dim]

        embedded = embedded.unsqueeze(1)
        # embedded = [batch size, 1, sent len, emb dim]

        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # conv_n = [batch size, n_filters, sent len - filter_sizes[n]]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, n_filters]

        cat = (torch.cat(pooled, dim=1))
        nums = self.mlp(nums)
        cat = self.dropout(torch.cat((cat, nums), dim=1))
        # cat = [batch size, n_filters * len(filter_sizes)]

        return self.fc(cat)

    def train_epoch(self, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0
        self.train()

        for batch in iterator:
            optimizer.zero_grad()
            predictions = self.forward(batch.text, batch.num)
            loss = criterion(predictions, batch.label)
            acc = categorical_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss/len(iterator), epoch_acc/len(iterator)

    def  evaluate_epoch(self, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        self.eval()
        with torch.no_grad():
            for batch in iterator:
                predictions = self.forward(batch.text, batch.num)

                loss = criterion(predictions, batch.label)
                acc = categorical_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss/len(iterator), epoch_acc/len(iterator)
