# import faulthandler
# faulthandler.enable()

from dataset import create_kfold_datasets
from config import MODEL_PARAMS, DATASET_PARAMS, SPLITS, MODEL_NAME, DATASET_NAME, MODEL_CLASSES
from train import train_kfold, train_process
from utils.io import read_file_locally


if __name__ == '__main__':
    # dataset = create_dataset(dataset_name='graph_dataset', save_folder='schiza')
    # train_dataset, test_dataset = simple_split(dataset, train_size=0.8)
    # model = GraphNetwork(MODEL_PARAMS['base']['hidden_dim'],
    #                      dataset.num_node_features,
    #                      dataset.num_classes)
    # model = train_process(model, train_dataset, test_dataset)
    

    # train_dataset = create_dataset(dataset_name='BrainNetCNN',
    #                                save_folder='schiza',
    #                                mode='train')
    # test_dataset = create_dataset(dataset_name='BrainNetCNN',
    #                               save_folder='schiza',
    #                               mode='validation')
    # model = BrainNetCNN(**MODEL_PARAMS['BrainNetCNN'],
    #                     example=train_dataset.X)
    # model = train_process(model, 'BrainNetCNN', train_dataset, test_dataset)

    kfold_splits = read_file_locally(SPLITS[DATASET_NAME])
    datasets = create_kfold_datasets(MODEL_NAME, 
                                     kfold_splits, 
                                     **DATASET_PARAMS[MODEL_NAME][DATASET_NAME])
    models = train_kfold(MODEL_CLASSES[MODEL_NAME],
                         MODEL_PARAMS[MODEL_NAME],
                         MODEL_NAME,
                         datasets)
    # model = MODEL_CLASSES[MODEL_NAME](**MODEL_PARAMS[MODEL_NAME], example=datasets[0][0].X)
    # model = train_process(model, MODEL_NAME, datasets[0][0], datasets[0][1])
    # print(model)