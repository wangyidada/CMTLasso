import os
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso,Lasso

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.Normalizer import NormalizerZscore
from FAE.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from FAE.Func.Metric import EstimateMetirc

from utils.CVGenerator import CVGenerator
from utils.SelectFeature import FeatureSelector
from utils.StatisticAUCs import statistic_aucs
from utils.myutils import MultiLabelTransform, Transform2SingleLabel
from STLasso import STLasso


class CMTLasso:
    def __init__(self, train_file, test_file, train_task=None,
                 save_path=None,  alpha_range=None, 
                 cv_folds=100, select_ratio=0.8, 
                 select_auc = 0.7, random_seed=12):
        self.train_file = train_file
        self.test_file = test_file
        self.train_task = train_task
        self.class_num=len(train_task)
        self.cv_folds = cv_folds
        self.save_path = save_path
        self.alpha_range = alpha_range
        self.select_ratio = select_ratio
        self.select_auc = select_auc
        self.random_seed=random_seed

    def makefolder(self, alpha):
        folder_name = 'alpha_' + str(alpha)
        self.figure_folder = os.path.join(self.save_path, folder_name,'Figures')
        os.makedirs(self.figure_folder, exist_ok=True)
        result_folder = os.path.join(self.save_path, folder_name, 'Results')
        os.makedirs(result_folder, exist_ok=True)
        self.merged_features_folder = os.path.join(self.save_path, folder_name, 'MergedFeatures')
        os.makedirs(self.merged_features_folder, exist_ok=True)

        self.cv_mt_folder = os.path.join(result_folder, 'CV_Multi_Task_Results')
        self.cv_st_folder = os.path.join(result_folder, 'CV_Single_Task_Results')


    def load_data(self):
        train_loader = DataContainer()
        train_loader.Load(self.train_file)
        return train_loader

    def Normalize(self, train_loader):
        normalizer = NormalizerZscore
        normalizer.Run(train_loader)
        normalized_train = normalizer.Transform(train_loader)
        normalizer.SaveInfo(self.save_path, normalized_train.GetFeatureName())
        normalizer.SaveNormalDataContainer(normalized_train, self.save_path, store_key='train')
        return normalized_train

    def ReduceDim(self, normalized_train):
        reducer = DimensionReductionByPCC(threshold=0.99)
        reducer.Run(normalized_train)
        dr_train = reducer.Transform(normalized_train)
        reducer.SaveInfo(self.save_path)
        reducer.SaveDataContainer(dr_train, self.save_path, store_key='train')
        return dr_train

    def SavePrediction(self, case_name, label_value, pred_value, save_file, task):
        info = []
        nums = len(task)
        if nums > 1:
            store_info = ['ID']+[('label' + str(i)) for i in range(nums)] + [('pred' +str(i)) for i in range(nums)]
        else:
            store_info = ['ID', 'label', 'pred']

        for i in range(len(case_name)):
            name = case_name[i]
            if nums ==1:
                label = [label_value[i]]
                pred = [pred_value[i]]
            else:
                label = [label_value[i][j] for j in range(nums)]
                pred = [pred_value[i][j] for j in range(nums)]
            info.append([name] + label + pred)
        self.write2csv(save_file, info, column_names=store_info)

    def SaveModel(self, feature_names, alpha, classifier, model_folder, task):
        os.makedirs(model_folder, exist_ok=True)
        coef_file = os.path.join(model_folder, 'coefs.csv')
        intercept_file = os.path.join(model_folder,  'intercept.csv')
        alpha_file = os.path.join(model_folder,  'alpha.csv')

        coef_data = {'feature_names': feature_names}
        intercept_data = {}

        if len(task) == 1:
            coef_data.update({('coef_task'):classifier.coef_.tolist()})
            intercept_data.update({('intercept_task'):[classifier.intercept_]})

        else:
            for i in range(len(task)):
                coef_data.update({('coef_task' + str(i)):classifier.coef_[i].tolist()})
                intercept_data.update({('intercept_task' + str(i)):[classifier.intercept_[0]]})

        coef_df = pd.DataFrame(coef_data)
        coef_df.to_csv(coef_file, index=False)

        intercept_df = pd.DataFrame(intercept_data)
        intercept_df.to_csv(intercept_file, index=False)

        alpha_data = { 'alpha': [alpha]}
        alpha_df = pd.DataFrame(alpha_data)
        alpha_df.to_csv(alpha_file, index=False)

    def CV_MT(self, dr_data, alpha):
        case_name = dr_data.GetCaseName()
        data = dr_data.GetArray()
        label = dr_data.GetLabel()
        feature_name  = dr_data.GetFeatureName()
        self.CVTrainer(case_name, feature_name, data, label, alpha, self.cv_mt_folder, 
                       task=self.train_task, class_num=self.class_num, is_MT=True)
       
    def CV_ST(self, dr_data, alpha):
        case_name = dr_data.GetCaseName()
        data = dr_data.GetArray()
        label = dr_data.GetLabel()
        feature_name  = dr_data.GetFeatureName()
        label = MultiLabelTransform(dr_data.GetLabel(), classes=len(self.train_task))

        for i, task in enumerate(self.train_task):
            print(i, task)
            single_label = np.array(label[:, i], dtype=np.int16)   
            save_folder = os.path.join(self.cv_st_folder, task)
            self.CVTrainer(case_name, feature_name, data, single_label, alpha, save_folder, 
                           task=[task], class_num=2,  is_MT=False)

    def CVTrainer(self, case_name, feature_name, data, label, alpha, result_folder, class_num, task, is_MT=True):
        cv_generator = CVGenerator(case_name, data, label, class_num=class_num, fold_num=self.cv_folds)
        for fold_index, (train_data, val_data) in enumerate(cv_generator.cv_split()):
            if is_MT:
                classifier = MultiTaskLasso(alpha=alpha, random_state=self.random_seed)
                train_label_data = MultiLabelTransform(train_data[2], classes=len(self.train_task))
                val_label_data = MultiLabelTransform(val_data[2], classes=len(self.train_task))
            else:
                classifier = Lasso(alpha=alpha, random_state=self.random_seed)
                train_label_data = train_data[2]
                val_label_data = val_data[2]

            classifier.fit(train_data[1], train_label_data)
            train_pred = classifier.predict(train_data[1])
            val_pred = classifier.predict(val_data[1])
        
            save_folder= os.path.join(result_folder, 'fold_'+str(fold_index))
            os.makedirs(save_folder, exist_ok=True)

            self.SaveModel(feature_name, alpha, classifier, os.path.join(save_folder, 'model'), task=task)
            self.SavePrediction(train_data[0], train_label_data, train_pred,  os.path.join(save_folder,'train_pred.csv'), task=task)
            self.SavePrediction(val_data[0], val_label_data, val_pred, os.path.join(save_folder, 'val_pred.csv'), task=task)
            self.CV_Evaluation(train_label_data, train_pred, val_label_data, val_pred, task, save_folder)

    def CV_Evaluation(self, train_label, train_pred, val_label, val_pred, task, save_folder):
        metric = {}
        preds = [train_pred, val_pred]
        labels = [train_label, val_label]
        key_words = ['cv_train' , 'cv_val']
        for key_word, label, pred in zip(key_words, labels, preds):
            self.Evaluate(label, pred, key_word=key_word, metric=metric, task=task)
            self.SaveMetric(metric, os.path.join(save_folder, 'evaluation_metrix.csv'))

    def Evaluate(self, label, pred, key_word, metric, task):
        label = np.asarray(label, dtype=np.uint8)
        pred = np.asarray(pred, dtype=np.float32)
        for i, t in enumerate(task):
            if len(task) > 1:
                task_eval = EstimateMetirc(pred[:, i], label[:, i], key_word + '_task_'+ t)
                metric.update(task_eval)
            else:
                task_eval = EstimateMetirc(pred, label, key_word + '_task_'+ t)
                metric.update(task_eval)
        return metric

    def write2csv(self, file, values, column_names=None):
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if column_names is not None:
                writer.writerow(column_names)
            writer.writerows(values)

    def SaveMetric(self, metrix, save_file):        
        df = pd.DataFrame(list(metrix.items()))
        df.columns = ['metrix', 'value']
        df.to_csv(save_file, index=False)

    def MergeFeatures(self):
        statistic_aucs(self.cv_mt_folder, self.cv_st_folder, self.train_task, self.figure_folder)

        FS = FeatureSelector(self.cv_mt_folder, self.cv_st_folder, tasks=self.train_task, 
                             fold_num=self.cv_folds, select_ratio=self.select_ratio,
                             select_auc=self.select_auc, save_folder=self.figure_folder)
        combined_features_names = FS.Run()

        self.SaveMergeFeatures(combined_features_names, self.train_file, self.merged_features_folder, key = 'train')
        self.SaveMergeFeatures(combined_features_names, self.test_file,  self.merged_features_folder, key = 'test')

    def SaveMergeFeatures(self, combined_features_names, feature_file, save_folder, key):
        for i, task in enumerate(self.train_task):
            save_folder_single_task = os.path.join(save_folder, task)
            os.makedirs(save_folder_single_task, exist_ok=True)
            column_names = ['CaseName', 'label'] + combined_features_names[i]
            df = pd.read_csv(feature_file)[column_names]
            df = Transform2SingleLabel(df, i) 
            save_file = os.path.join(save_folder_single_task, key + '_MergedFeature.csv')
            df.to_csv(save_file, index=None)

    def FinalSTLasso(self, task):
        train_file = os.path.join(self.merged_features_folder, task, 'train_MergedFeature.csv')
        test_file = os.path.join(self.merged_features_folder, task, 'test_MergedFeature.csv')
        save_path = os.path.join(self.merged_features_folder, task)

        STL = STLasso(train_file, test_file, save_path, 
                  min_alpha=-4, max_alpha=1, 
                  steps=100, cv_folds=5)
        STL.Run()
   
    def Run(self):
        train_loader = self.load_data()
        normalized_train = self.Normalize(train_loader)
        dr_train= self.ReduceDim(normalized_train)

        for alpha in alpha_range:
            self.makefolder(alpha)
            self.CV_ST(dr_train, alpha)
            self.CV_MT(dr_train, alpha)            
            self.MergeFeatures()
            self.FinalSTLasso(self.train_task[0])
            self.FinalSTLasso(self.train_task[1])


if __name__ == '__main__':
    train_task=['IDH', 'EP']
    test_task= ['IDH', 'EP']
    alpha_range = [0.001, 0.01, 0.1]

    data_root = r'C:\Users\wyd\Desktop\EPMTL\CMT\file\CMT'
    train_file = os.path.join(data_root, 'train_features.csv')
    test_file = os.path.join(data_root, 'test_features.csv')
    save_path = data_root

    CMTL = CMTLasso(train_file, test_file, train_task, save_path, 
                   alpha_range=alpha_range, cv_folds=100, 
                   select_ratio=0.8, select_auc = 0.7, random_seed=6)
    CMTL.Run()




    












