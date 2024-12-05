import os
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLassoCV, MultiTaskLasso

from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.Normalizer import NormalizerZscore
from FAE.FeatureAnalysis.DimensionReduction import DimensionReductionByPCC
from FAE.Func.Metric import EstimateMetirc

from utils.myutils import MultiLabelTransform
from utils.plt_figures import plot_error, plot_coefs, Plot_ROCs


class MTLasso:
    def __init__(self, train_file, test_file, save_path, class_num=4,
                 min_alpha=-4, max_alpha=0, steps=10, cv_folds=5, random_seed=12):
        self.train_file = train_file
        self.test_file = test_file
        self.class_num=class_num
        self.cv_folds = cv_folds
        self.random_seed=random_seed
        self.alpha_range = np.logspace(min_alpha, max_alpha, steps)
        self.save_path = save_path
        self.makefolder(save_path)

    def makefolder(self, save_path):
        self.OneSE_model_folder = os.path.join(save_path, 'Models', 'OneSE')
        self.OneSE_result_folder = os.path.join(save_path,'Results', 'OneSE')
        self.CV_model_folder = os.path.join(save_path, 'Models', 'CV')
        self.CV_result_folder = os.path.join(save_path,'Results', 'CV') 

        os.makedirs(self.OneSE_result_folder, exist_ok=True) 
        os.makedirs(self.OneSE_model_folder, exist_ok=True) 
        os.makedirs(self.CV_result_folder, exist_ok=True) 
        os.makedirs(self.CV_model_folder, exist_ok=True) 

        self.figure_folder = os.path.join(save_path, 'Figures')
        os.makedirs(self.figure_folder, exist_ok=True) 

    def load_data(self):
        train_loader = DataContainer()
        train_loader.Load(self.train_file)
        test_loader = DataContainer()
        test_loader.Load(self.test_file)
        return train_loader, test_loader

    def Normalize(self, train_loader, test_loader):
        normalizer = NormalizerZscore
        normalizer.Run(train_loader)
        normalized_train = normalizer.Transform(train_loader)
        normalized_test = normalizer.Transform(test_loader)
        normalizer.SaveInfo(self.save_path, normalized_train.GetFeatureName())
        normalizer.SaveNormalDataContainer(normalized_train, self.save_path, store_key='train')
        normalizer.SaveNormalDataContainer(normalized_test, self.save_path, store_key='test')
        return normalized_train, normalized_test

    def ReduceDim(self, normalized_train, normalized_test):
        reducer = DimensionReductionByPCC(threshold=0.99)
        reducer.Run(normalized_train)
        dr_train = reducer.Transform(normalized_train)
        dr_test = reducer.Transform(normalized_test)
        reducer.SaveInfo(self.save_path)
        reducer.SaveDataContainer(dr_train, self.save_path, store_key='train')
        reducer.SaveDataContainer(dr_test, self.save_path, store_key='test')
        return dr_train, dr_test


    def TrainTest(self, train_value, test_value, alpha, result_folder):
        classifier = MultiTaskLasso(alpha=alpha, random_state=self.random_seed)
        classifier.fit(train_value[1], train_value[2])

        train_pred = classifier.predict(train_value[1])
        test_pred = classifier.predict(test_value[1])

        train_file = os.path.join(result_folder,'train_pred.csv')
        test_file = os.path.join(result_folder, 'test_pred.csv')

        self.SavePrediction(train_value, train_pred, train_file)
        self.SavePrediction(test_value, test_pred, test_file)

        metrix = {}
        for i in range(self.class_num):
            train_task_eval = EstimateMetirc(train_pred[:, i], train_value[2][:, i], 'train_task'+str(i))
            metrix.update(train_task_eval)
            test_task_eval = EstimateMetirc(test_pred[:, i], test_value[2][:, i], 'test_task'+str(i))
            metrix.update(test_task_eval)

        return metrix, classifier


    def SavePrediction(self, value, pred_value, save_file):
        info = []
        store_info = ['ID']+[('label' +str(i)) for i in range(self.class_num)] + [('pred' +str(i)) for i in range(self.class_num)]

        for i in range(len(value[0])):
            name = value[0][i]
            label = [value[2][i][j] for j in range(self.class_num)]
            pred = [pred_value[i][j] for j in range(self.class_num)]
            info.append([name] +label+ pred)
        self.write2csv(save_file, info, column_names=store_info)


    def SaveModel(self, feature_names, classifier, alpha, model_folder):
        coef_file = os.path.join(model_folder, 'coefs.csv')
        intercept_file = os.path.join(model_folder,  'intercept.csv')
        alpha_file = os.path.join(model_folder,  'alpha.csv')

        coef_data = { 'feature_names': feature_names}
        intercept_data = {}

        for i in range(self.class_num):
            coef_data.update({('coef_task' + str(i)):classifier.coef_[i].tolist()})
            intercept_data.update({('intercept_task' + str(i)):[classifier.intercept_[0]]})

        coef_df = pd.DataFrame(coef_data)
        coef_df.to_csv(coef_file, index=False)
        intercept_df = pd.DataFrame(intercept_data)
        intercept_df.to_csv(intercept_file, index=False)

        alpha_data = { 'alpha': [alpha]}
        alpha_df = pd.DataFrame(alpha_data)
        alpha_df.to_csv(alpha_file, index=False)


    def GetAplha(self, dr_train):
        train_data= dr_train.GetArray()
        train_label = MultiLabelTransform(dr_train.GetLabel(), classes=2)
        classifier = MultiTaskLassoCV(cv=self.cv_folds, alphas=self.alpha_range, random_state=self.random_seed)
        classifier.fit(train_data, train_label)
        best_alpha = self.GetOneSE(classifier.alphas_, classifier.mse_path_)
        coefs = MultiTaskLassoCV.path(train_data, train_label, alphas=self.alpha_range, random_state=self.random_seed)[1].T
        plot_error(classifier.alphas_, classifier.mse_path_, classifier.alpha_, self.figure_folder)
        plot_coefs(classifier.alphas_, coefs, best_alpha, self.figure_folder)
        return best_alpha, classifier.alpha_


    def Classify(self, dr_train, dr_test, alpha, model_folder, result_folder):
        train_case_name, test_case_name = dr_train.GetCaseName(), dr_test.GetCaseName()
        train_data, test_data = dr_train.GetArray(), dr_test.GetArray()

        train_label = MultiLabelTransform(dr_train.GetLabel(), classes=2)
        test_label = MultiLabelTransform(dr_test.GetLabel(), classes=2)

        train_value = [train_case_name, train_data, train_label]
        test_value = [test_case_name, test_data, test_label]

        metrix, classifier = self.TrainTest(train_value, test_value, alpha, result_folder)
        self.SaveModel(dr_train.GetFeatureName(), classifier, alpha, model_folder)

        df = pd.DataFrame(list(metrix.items()))
        file = os.path.join(result_folder, 'results.csv')
        df.columns = ['metrix', 'value']
        df.to_csv(file, index=False)
        
        pred_files = [os.path.join(result_folder, 'train_pred.csv'), os.path.join(result_folder, 'test_pred.csv')]
        self.PlotROCs(pred_files, result_folder)
    

    def PlotROCs(self, pred_files, result_folder):
        Plot_ROCs(pred_files,  pred_name='pred0', label_name='label0', save_file=os.path.join(result_folder, 'Task0ROC.jpg'))
        Plot_ROCs(pred_files,  pred_name='pred1', label_name='label1', save_file=os.path.join(result_folder, 'Task1ROC.jpg'))


    def GetOneSE(self, alphas, mse):
        best_alpha = 0
        mse_mean =list(np.mean(mse, axis=1))
        mse_std = list(np.std(mse, axis=1))
        one_se = min(mse_mean) + mse_std[mse_mean.index(min(mse_mean))]

        for index in range(len(mse_mean)):
            if mse_mean[index] <= one_se:
                best_alpha = alphas[index]
                break
        return best_alpha

        
    def write2csv(self, file, values, column_names=None):
        with open(file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if column_names is not None:
                writer.writerow(column_names)
            writer.writerows(values)


    def Run(self):
        train_loader, test_loader = self.load_data()
        normalized_train, normalized_test = self.Normalize(train_loader, test_loader)
        dr_train, dr_test = self.ReduceDim(normalized_train, normalized_test)
        [best_alpha, cvalpha] = self.GetAplha(dr_train)
        print('CV alpha is', cvalpha)
        print('Best alpha is', best_alpha)

        self.Classify(dr_train, dr_test, best_alpha, self.OneSE_model_folder, self.OneSE_result_folder)
        self.Classify(dr_train, dr_test, cvalpha, self.CV_model_folder, self.CV_result_folder)



if __name__ == '__main__':
    train_task=['IDH', 'EP']
    train_file = r'C:\Users\wyd\Desktop\EPMTL\CMT\file\MT\train_features.csv'
    test_file = r'C:\Users\wyd\Desktop\EPMTL\CMT\file\MT\test_features.csv'
    save_path = r'C:\Users\wyd\Desktop\EPMTL\CMT\file\MT'
    MTL = MTLasso(train_file, test_file, save_path, class_num=2,
                    min_alpha=-4, max_alpha=1, 
                    steps=100, cv_folds=5)
    MTL.Run()
   
   



    






















    




















