import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

DATABASE_NAME='v1pv2.csv'
TRAINING_RATIO=0.6
VALIDATION_RATIO=0.2 #test ratio is determined by those two ratios as 1-training_ratio-validation_ratio

class SimpleDataset(Dataset):
        def __init__(self, csv_file=DATABASE_NAME,type='train'):
                #type can be 'train','val' or 'test'

                #read the database of examples, and divide it into training,validation and test sets
                db=pd.read_csv(DATABASE_NAME)
                num_examples=len(db)
                training_index=round(num_examples*TRAINING_RATIO)
                validation_index=training_index+round(num_examples*VALIDATION_RATIO)
                training=db[0:training_index]
                validation=db[training_index:validation_index]
                test=db[validation_index:]
                if type=='train':
                        self.db=training
                elif type=='val':
                        self.db=validation
                else:
                        assert(type=='test')
                        self.db=test
        def __len__(self):
                return len(self.db)
        def __getitem__(self, idx):
                # from pdb_clone import pdb
                # pdb.set_trace()
                return {'example':self.db.ix[idx,:-1].as_matrix().astype('float'),'target':np.array(self.db.ix[idx,-1]).reshape(1,)}