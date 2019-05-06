import torch
from torchtext import data
import time
from pathlib import Path


def cross_val_score(Model, model_kwargs, model_path,
                    custom_embeddings, vocab_kwargs,
                    data_path,
                    label_column, text_column, other_fields,
                    process_text, process_labels,
                    Optimizer, optimizer_kwargs, criterion,
                    batch_size, n_epochs, writer, device,
                    ):
    p = Path(data_path)
    n_files = len(list(p.glob('*.json')))
    # тут мы полагаем что на каждый фолд должно приходится 2 файла: test и train
    assert n_files % 2 == 0
    n_splits = n_files // 2
    # будем поддерживать масивы с accuracy на валидации для последней эпохи, и accuracy на эпохе с лучшим лоссом
    best_accuracy = []
    final_accuracy = []
    for fold in range(n_splits):

        # всем используемым моделям нужны поля с текстом и целевым лейблом
        TEXT = data.Field(**process_text)
        LABEL = data.LabelField(**process_labels)
        fields = {label_column: ('label', LABEL), text_column: ('text', TEXT)}
        # некоторые модели требуют дополнительные поля, их опредялем в вызывающем контексте
        fields.update(other_fields)


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

        input_dim = len(TEXT.vocab)
        output_dim = len(LABEL.vocab)
        pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
        model = Model(input_dim, output_dim, pad_idx=pad_idx, **model_kwargs)

        if custom_embeddings is not None:
            embeddings = TEXT.vocab.vectors
            model.embedding.weight.data.copy_(embeddings)

        # явно зануляем ембеддинг для <pad>
        model.embedding.weight.data[pad_idx] = torch.zeros(model_kwargs['embedding_dim'])
        optimizer = Optimizer(model.parameters(), **optimizer_kwargs)
        model = model.to(device)
        criterion = criterion.to(device)

        train_iterator, test_iterator = data.BucketIterator.splits(
            (train_data, test_data),
            batch_size=batch_size,
            sort_key=lambda ex: len(ex.text),
            sort_within_batch=True,
            device=device)

        best_valid_acc, final_valid_acc = train_model(model, train_iterator, test_iterator,
                                                      optimizer, criterion, model_path + f'_{fold}',
                                                      n_epochs=n_epochs, comment=f'fold_{fold}', writer=writer)

        best_accuracy.append(best_valid_acc)
        final_accuracy.append(final_valid_acc)

    return best_accuracy, final_accuracy


def train_model(model, train_iterator, valid_iterator,
                optimizer, criterion, model_path,
                n_epochs, comment, writer):

    best_valid_acc = 0.0
    best_valid_loss = float('inf')
    train_start = time.time()

    train_loss, train_acc = model.evaluate_epoch(train_iterator, criterion)
    valid_loss, valid_acc = model.evaluate_epoch(valid_iterator, criterion)

    # пишем в tensorboard лосс и аккураси до начала обучения
    writer.add_scalars(f'metrics_{comment}/loss',
                       {'train': train_loss,
                        'valid': valid_loss},
                       0
                       )
    writer.add_scalars(f'metrics_{comment}/accuracy',
                       {'train': train_acc,
                        'valid': valid_acc},
                       0
                       )
    for epoch in range(n_epochs):
        train_loss, train_acc = model.train_epoch(train_iterator, optimizer, criterion)
        valid_loss, valid_acc = model.evaluate_epoch(valid_iterator, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), model_path)


        writer.add_scalars(f'metrics_{comment}/loss',
                           {'train': train_loss,
                            'valid': valid_loss},
                           epoch+1
                           )
        writer.add_scalars(f'metrics_{comment}/accuracy',
                           {'train': train_acc,
                            'valid': valid_acc},
                           epoch+1
                           )

    train_end = time.time()
    train_mins, train_secs = epoch_time(train_start, train_end)

    writer.add_text(f'Model description, {comment}', str(model))
    writer.add_text(f'Training time, {comment}', f'{train_mins}m {train_secs}s')
    writer.close()

    return best_valid_acc, valid_acc

# дальше идет набор вспомогательных функций

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum() / torch.FloatTensor([y.shape[0]])

# для cnn  нам необходимо чтобы все тексты были не меньше определенной длины
# поэтому используем вот такой генератор функций дополняющих предложение до требуемой длины
# используется в препроцессенге поля
def make_padder(min_len=4):
    def pad(seq):
        if len(seq) >= min_len:
            return seq
        else:
            return seq + ['<unk>'] * (min_len - len(seq))
    return pad

# не совсем удачное название, поскольку для замера эпох я ее больше не использую
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

