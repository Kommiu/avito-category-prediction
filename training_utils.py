import torch
from torchtext import data
import time
from pathlib import Path


def cross_val_score(Model, model_kwargs, model_name,
              custom_embeddings, vocab_kwargs,
              data_path,
              label_field, text_field, other_fields,
              process_text, process_labels,
              Optimizer, optimizer_kwargs, criterion,
              batch_size, n_epochs, writer, device,
              ):
    p = Path(data_path)
    n_files = len(list(p.glob('*.json')))
    assert n_files % 2 == 0
    n_splits = n_files // 2
    accuracy = []
    for fold in range(n_splits):

        TEXT = data.Field(**process_text)
        LABEL = data.LabelField(**process_labels)
        fields = {label_field: ('label', LABEL), text_field: ('text', TEXT)}
        fields.update(other_fields)

        print(f'\nFold-{fold}:')

        train_data = data.TabularDataset(
            path=Path(data_path, f'train_{fold}.json'),
            format='json',
            fields=fields,
        )
        test_data = data.TabularDataset(
            path=Path(data_path, f'test_{fold}.json'),
            format='json',
            fields=fields,
        )

        TEXT.build_vocab(train_data, vectors=custom_embeddings, **vocab_kwargs)
        LABEL.build_vocab(train_data)
        print(f'Vocab size: {len(TEXT.vocab)}')
        print(f'Number of classes: {len(LABEL.vocab)}')

        input_dim = len(TEXT.vocab)
        output_dim = len(LABEL.vocab)
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        model = Model(input_dim, output_dim, pad_idx=pad_idx, **model_kwargs)

        if custom_embeddings is not None:
            embeddings = TEXT.vocab.vectors
            model.embedding.weight.data.copy_(embeddings)

        unk_idx = TEXT.vocab.stoi[TEXT.unk_token]

        model.embedding.weight.data[unk_idx] = torch.zeros(model_kwargs['embedding_dim'])
        model.embedding.weight.data[pad_idx] = torch.zeros(model_kwargs['embedding_dim'])
        optimizer = Optimizer(model.parameters(), **optimizer_kwargs)
        model = model.to(device)
        criterion = criterion.to(device)

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=batch_size,
            sort_key=lambda ex: len(ex.text),
            sort_within_batch=False,
            device=device)

        print('\tTraining now...')
        valid_acc = train_model(model, train_iterator, test_iterator,
                                optimizer, criterion, model_name + f'_{fold}',
                                n_epochs=n_epochs, comment=f'fold_{fold}', writer=writer)

        accuracy.append(valid_acc)

    return accuracy

def train_model(model, train_iterator, valid_iterator,
                optimizer, criterion, model_name,
                n_epochs, comment, writer=None):

    best_valid_loss = float('inf')
    train_start = time.time()

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = model.train_epoch(train_iterator, optimizer, criterion)
        valid_loss, valid_acc = model.evaluate_epoch(valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)

        print(f'\tEpoch: {epoch + 1} | Epoch Time: {epoch_mins}m {epoch_secs}:s ')

        if writer:
            writer.add_scalars(f'metrics_{comment}/loss',
                               {'train': train_loss,
                                'valid': valid_loss},
                               epoch
                               )
            writer.add_scalars(f'metrics_{comment}/accuracy',
                               {'train': train_acc,
                                'valid': valid_acc},
                               epoch
                               )

    train_end = time.time()
    train_mins, train_secs = epoch_time(train_start, train_end)

    if writer:
        writer.add_text(f'Model description, {comment}', str(model))
        writer.add_text(f'Training time, {comment}', f'{train_mins}m {train_secs}s')
        writer.close()

    return valid_acc
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text)

        loss = criterion(predictions, batch.label)

        acc = categorical_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text)

            loss = criterion(predictions, batch.label)

            acc = categorical_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
