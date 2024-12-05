import numpy as np
import random
from utils.myutils import MultiLabelTransform

class CVGenerator:
    def __init__(self, case_name, data, label, class_num=4,fold_num=20, mode='percentage', 
                 percentage=[0.8, 0.2], number=None, is_replace_sampling=True):
        self.case_name = case_name
        self.data = data
        self.label = label
        self.class_num = class_num
        self.percentage = percentage
        self.fold_num = fold_num
        self.mode = mode
        self.number= number
        self.is_replace_sampling = is_replace_sampling
    
    def split_index(self, total_number, random_seed):
        if self.mode == 'percentage':
            train_num = np.floor(total_number*self.percentage[0])
            val_num = total_number - train_num 
            number = [train_num, val_num]
            number = [np.int(x) for x in number]
        else:
            number = self.number
        total_index = range(total_number)
        random.seed(random_seed)
        train_index = random.sample(total_index, number[0])
        val_index = [x for x in total_index if x not in train_index]
        if self.is_replace_sampling:
            np.random.seed(random_seed)
            train_index = np.random.choice(train_index, len(train_index), replace=True)
            train_index = train_index.tolist()
        return train_index, val_index

    def get_index(self, random_seed):
        total_number = len(self.case_name)
        train_index, val_index = self.split_index(total_number, random_seed)
        train_data = [[self.case_name[i] for i in train_index], self.data[train_index, :],self.label[train_index]]
        val_data = [[self.case_name[i] for i in val_index], self.data[val_index, :], self.label[val_index]]
        return train_data, val_data

    def cv_split(self):
        cv_data = []
        for i in range(self.fold_num):
            random_seed = i
            train_data, val_data = self.get_index(random_seed=random_seed)
            cv_data.append([train_data, val_data])
        return cv_data
    

if __name__ == '__main__':
    case_name = ['1']*20+['2']*10
    data = np.random.normal(loc=0, scale=1, size=(30, 2))
    label = np.ones((30, 1), dtype=np.int)
    print(label[:, 0].shape)

    cv = CVGenerator(case_name, data, label, class_num=4, fold_num=3, is_replace_sampling=True)
    cv_data = cv.cv_split()
   

   
   
