import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch.autograd import Variable


def train_kfold(model_class, model_params, model_name, datasets):
    models = []
    losses = []
    train_metrics = []
    val_metrics = []
    
    for i, (train, val) in enumerate(datasets):
        print(f'Training fold #{i+1}')
        model = model_class(**model_params, example=train.X)
        trained_model, metrics = train_process(model, model_name, train, val, return_stats=True)
        models.append(trained_model)
        losses.append(metrics['loss'][-1])
        train_metrics.append(metrics['train_metrics'][-1])
        val_metrics.append(metrics['val_metrics'][-1])
    
    print(f'Trained {len(datasets)} folds')
    print(f'Train metrics for each fold: {train_metrics}\nMean train metric: {np.mean(train_metrics):.4f}')
    print(f'Validation metrics for each fold: {val_metrics}\nMean validation metric: {np.mean(val_metrics):.4f}')

    return models


def train_process(model, model_name, train_dataset, test_dataset, return_stats=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=0.00001, momentum=0.9,
                                nesterov=True,
                                weight_decay=0.0005)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # loss_fn = torch.nn.NLLLoss()
    loss_fn = torch.nn.MSELoss()
    losses = []
    train_metrics = []
    val_metrics = []
    for epoch in range(0, 100):
        loss = train_func[model_name](model, loss_fn, device, train_loader,
                                      optimizer)
        train_result = eval_func[model_name](model, device, train_loader)
        val_result = eval_func[model_name](model, device, test_loader)
        losses.append(loss)
        train_metrics.append(train_result)
        val_metrics.append(val_result)

        print(f'Epoch: {epoch + 1:02d}, '
              f'Loss: {loss:.8f}, '
              f'Train: {100 * train_result:.2f}%, '
              f'Test: {100 * val_result:.2f}%')
    if return_stats:
        return model, {'loss':losses, 'train_metrics': train_metrics, 'val_metrics': val_metrics}
    return model


def eval(model, model_name, data_loader, device):
    return eval_func[model_name](model, device, data_loader)


def _train_base(model, loss_fn, device, data_loader, optimizer):
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


def _train_brainnet(model, loss_fn, device, data_loader, optimizer):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()

    return running_loss / len(data_loader.dataset)


def _eval_base(model, device, loader):
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


def _eval_brainnet(model, device, data_loader):
    test_acc = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            test_acc += (np.argmax(outputs.cpu().numpy(), axis=1) ==
                         np.argmax(targets.cpu().numpy(), axis=1)).sum()

    test_acc /= len(data_loader.dataset)
    return test_acc


train_func = {
    'base': _train_base,
    'BrainNetCNN': _train_brainnet
}

eval_func = {
    'base': _eval_base,
    'BrainNetCNN': _eval_brainnet
}
