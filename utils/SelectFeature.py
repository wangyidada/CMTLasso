import os
import numpy as np
import pandas as pd
from collections import Counter
from utils.Visualization import plot_bar, plotCompareFeautures


class FeatureSelector:
    def __init__(self, multi_task_folder, single_task_folder, tasks, fold_num, select_ratio=0.8, select_auc = 0.7, save_folder=None):
        self.multi_task_folder = multi_task_folder
        self.single_task_folder = single_task_folder
        self.select_ratio = select_ratio
        self.select_auc = select_auc
        self.fold_num = fold_num
        self.tasks = tasks
        self.save_folder = save_folder
    
    def get_nonezero_features(self, file_path):
        weighted_features = {}
        weights = []
        df = pd.read_csv(file_path)
        columms = df.columns.values
        features = df['feature_names'].tolist()
        for col in columms[1:]:
            weights.append(abs(df[col].values))
        
        weights = np.array(weights, dtype=np.float32)
        weights_sum = np.sum(weights, axis=0)
        weights_nonezero_index = list(np.where(weights_sum!=0)[0])
        for i in weights_nonezero_index:
            weighted_features[features[i]] = weights_sum[i]
        return weighted_features

    def sortedfeature(self, folder, tasks):
        weights_total = {}
        for file in os.listdir(folder):
            auc_file = os.path.join(folder, file, 'evaluation_metrix.csv')
            df = pd.read_csv(auc_file)
            if len(tasks)==1:
                name = 'cv_val_task_' + tasks[0] + '_auc'
                val_auc_value = np.array(df[df['metrix']==name]['value'].values, dtype=np.float32)
            else:
                val_aucs = []
                for t in tasks:
                    name = 'cv_val_task_' + t + '_auc'
                    val_auc = df[df['metrix']==name]['value'].values
                    val_aucs.append(val_auc[0])
                val_auc_value =  np.mean(np.array(val_aucs,  dtype=np.float32))

            if val_auc_value < self.select_auc:
                continue

            file_path = os.path.join(folder, file, 'model', 'coefs.csv')
            weighted_features = self.get_nonezero_features(file_path)        
            sorted_features=sorted(weighted_features.items(), key=lambda x:x[1],reverse=True)
            weights_total[file] = dict(sorted_features)
        return weights_total

    def selectfeaturebytimes(self, weights_total):
        features_total= []
        n = int(np.round(self.fold_num* self.select_ratio))
        for name, value in weights_total.items():
            features_total = features_total + (list(set(value)))

        select_featuresbytimes = dict(Counter(features_total))
        select_featuresbytimes=sorted(select_featuresbytimes.items(), key=lambda x:x[1],reverse=True)
        condition_func = lambda x: x[1] > n
        features_select = dict(select_featuresbytimes)
        features_select_new = {k: v for k, v in features_select.items() if condition_func((k, v))}
        return features_select_new
        
    def MergeFeatures(self, features1, features2):
        total_feature = list(dict(features1).keys())+ list(dict(features2).keys())
        return list(set(total_feature))

    def Run(self):
        total_features = []
        select_featuresbytimes_MT = self.selectfeaturebytimes(self.sortedfeature(self.multi_task_folder, tasks = self.tasks))
        plot_bar(select_featuresbytimes_MT, self.save_folder, title='MultiTask')

        for task in self.tasks:
            single_folder_child = os.path.join(self.single_task_folder, task)
            select_featuresbytimes_ST = self.selectfeaturebytimes(self.sortedfeature(single_folder_child, tasks = [task]))
            total_featurefortask = self.MergeFeatures(select_featuresbytimes_MT.copy(), select_featuresbytimes_ST.copy())
            total_features.append(total_featurefortask)
            plot_bar(select_featuresbytimes_ST, self.save_folder, title=task)
            plotCompareFeautures(select_featuresbytimes_MT.copy(), select_featuresbytimes_ST.copy(), label=task, save_folder=self.save_folder)
        
        return total_features



