import torch
from torch_geometric.data import DataLoader


def train_process(model, train_dataset, test_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    loss_fn = torch.nn.NLLLoss()
    losses = []

    for epoch in range(0, 30):
        loss = train(model, loss_fn, device, train_loader, optimizer)
        train_result = eval(model, device, train_loader)
        test_result = eval(model, device, test_loader)
        losses.append(loss)

        print(f'Epoch: {epoch + 1:02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_result:.2f}%, '
              f'Test: {100 * test_result:.2f}%')

    return model


def train(model, loss_fn, device, data_loader, optimizer):
    """ Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    loss = 0

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
    return loss.item()


def eval(model, device, loader):
    """ Calculate accuracy for all examples in a DataLoader.

    Parameters:
    model (nn.Module): Model to be evaluated
    device (torch.Device): Device used for training
    loader (torch.utils.data.DataLoader): DataLoader containing examples to test
    """
    model.eval()
    cor = 0
    tot = 0

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            pred = torch.argmax(model(batch), 1)
        y = batch.y
        cor += (pred == y).sum()
        tot += pred.shape[0]

    return cor / tot
