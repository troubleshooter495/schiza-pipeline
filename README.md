# schiza-pipeline
Pipeline for training and testing interpretative models in neurology and psychiatry.

All work is done in the scope of "Development of clinically inspired ML and DL methods to construct interpretable predictive models in neurology and psychiatry" graduation project.


Higher School of Economics, The Skolkovo Institute of Science and Technology, 2023


### TODO
* move train/test arguments to config.py
* solve issue with BrainNetCNN datasets (use one function to create dataset and split it later)
* add BrainNetCNN metrics to eval
* add validation functions
* add cross validation (including loading from preset split)
* fix random seed
* add logging to file and checkpoints to training
* add loading pretrained model
* add save_file_locally() to io.py