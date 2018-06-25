### Model Training
Contains the model structure definition and training process here. 

#### [*data_preprocess.py*](./data_preprocess.py)
Dump the images and corresponding one-hot tags to a binary file. It can speed the model training process.

#### [*data_loader.py*](./data_loader.py)
Implementation the methods for GAN's data loading during the training stage inheriting from torch.utils.data.Dataset.

#### [*gan.py*](./gan.py)
The strategy for training GAN. Read codes for more details.

#### [*networks/discriminator.py*](./networks/discriminator.py)
Implementation of the discriminator network following the structure described in the paper with a faintly modification.

#### [*networks/generator.py*](./networks/generator.py)
Implementation of the generator network following the structure described in the paper with a faintly modification.

#### [*utils.py*](./utils.py)
Useful functions here.
