

def simple_split(data, train_size=0.8):
    train_share = int(len(data) * train_size)
    train_dataset = data[:train_share]
    test_dataset = data[train_share:]

    return train_dataset, test_dataset