import os
import pandas as pd
import numpy as np
from FAE.DataContainer.DataContainer import DataContainer
from FAE.FeatureAnalysis.Normalizer import NormalizerZscore
from sklearn.metrics import auc, roc_curve


def Normalize(train_file, test_file, save_path):
    loader = DataContainer()
    loader.Load(train_file)
    test_loader = DataContainer()
    test_loader.Load(test_file)
    normalizer = NormalizerZscore
    normalizer.Run(loader)
    normalized_test = normalizer.Transform(test_loader)
    normalizer.SaveNormalDataContainer(normalized_test, save_path, store_key='EXtest')


def load_ST_model(model_folder, test_file, pred_file, t=None):
    coef_file = os.path.join(model_folder, 'coefs.csv')  
    intercept_file = os.path.join(model_folder,'intercept.csv')

    df_feature = pd.read_csv(test_file)
    ids = df_feature.iloc[:, 0].values
    label_value = df_feature['label'].values.tolist()

    coef_df = pd.read_csv(coef_file)
    if t is None:    
        coef_df.drop(index = coef_df[coef_df['coef']==0].index, inplace=True)
    else:
        coef_df.drop(index = coef_df[abs(coef_df['coef'])<=t].index, inplace=True)

    features_names = coef_df['feature_names'].values.tolist()
    coef_values = coef_df['coef'].values
    intercept = pd.read_csv(intercept_file)['intercept'].values[0]

    pred = []

    for f, coef in zip(features_names, coef_values):
        feature = np.array(df_feature[f].values, dtype=np.float32)
        pred.append(feature*coef)
    
    pred_final =  np.sum(np.array(pred, dtype=np.float32), axis=0) + intercept
    df_pred = pd.DataFrame({'ID': list(ids), 'Label': label_value, 'Pred': list(pred_final)} )
    df_pred.to_csv(pred_file, index=None)


def load_MT_model(model_folder, test_file, pred_file, task='coef_task0', ip ='intercept_task0', t=None):
    coef_file = os.path.join(model_folder, 'coefs.csv')  
    intercept_file = os.path.join(model_folder,'intercept.csv')

    df_feature = pd.read_csv(test_file)
    ids = df_feature.iloc[:, 0].values
    label_value = df_feature['label'].values.tolist()

    coef_df = pd.read_csv(coef_file)
    if t is None:    
        coef_df.drop(index = coef_df[coef_df[task]==0].index, inplace=True)
    else:
        coef_df.drop(index = coef_df[abs(coef_df[task])<=t].index, inplace=True)

    features_names = coef_df['feature_names'].values.tolist()
    coef_values = coef_df[task].values
    intercept = pd.read_csv(intercept_file)[ip].values[0]

    pred = []

    for f, coef in zip(features_names, coef_values):
        feature = np.array(df_feature[f].values, dtype=np.float32)
        pred.append(feature*coef)
    
    pred_final =  np.sum(np.array(pred, dtype=np.float32), axis=0) + intercept

    df_pred = pd.DataFrame({'ID': list(ids), 'Label': label_value, 'Pred': list(pred_final)} )
    df_pred.to_csv(pred_file, index=None)

def get_feature_names(coef_file, task= 'coef'):
    coef_df = pd.read_csv(coef_file)    
    coef_df.drop(index = coef_df[coef_df[task]==0].index, inplace=True)
    features_names = coef_df['feature_names'].values.tolist()
    return features_names

def select_features(train_file, test_file, save_train_file, save_test_file, feature_names):
    fn = ['CaseName', 'label'] + feature_names
    train_df = pd.read_csv(train_file)[fn]
    test_df = pd.read_csv(test_file)[fn]
    train_df.to_csv(save_train_file, index=None)
    test_df.to_csv(save_test_file, index=None)


def Pred(train_file, model_folder, ex_folder):
    coef_file = os.path.join(model_folder, 'coefs.csv')
    test_file = os.path.join(ex_folder, 'tcia_feature.csv')
    save_train_file = os.path.join(ex_folder, 'select_train_features.csv')
    save_test_file = os.path.join(ex_folder, 'select_test_features.csv')

    feature_names = get_feature_names(coef_file)
    select_features(train_file, test_file, save_train_file, save_test_file, feature_names)
    Normalize(save_train_file, save_test_file, ex_folder)

    result_folder = os.path.join(ex_folder, 'result')
    os.makedirs(result_folder, exist_ok=True)
    pred_file = os.path.join(result_folder, 'Pred_Exteral.csv')   
    test_file = os.path.join(ex_folder, 'Zscore_normalized_EXtest_feature.csv')

    load_ST_model(model_folder, test_file, pred_file)


def Pred_MT(train_file, model_folder, ex_folder):
    coef_file = os.path.join(model_folder, 'coefs.csv')

    test_file = os.path.join(ex_folder, 'tcia_feature.csv')
    save_train_file = os.path.join(ex_folder, 'select_train_features.csv')
    save_test_file = os.path.join(ex_folder, 'select_test_features.csv')

    feature_names = get_feature_names(coef_file, 'coef_task0')
    select_features(train_file, test_file, save_train_file, save_test_file, feature_names)
    Normalize(save_test_file, save_test_file, ex_folder)

    result_folder = os.path.join(ex_folder, 'result')
    os.makedirs(result_folder, exist_ok=True)
    pred_file = os.path.join(result_folder, 'Pred_Exteral.csv')   
    test_file = os.path.join(ex_folder, 'Zscore_normalized_EXtest_feature.csv')

    load_MT_model(model_folder, test_file, pred_file)



if __name__ == '__main__':
    task =r'IDH'
    train_file =r'/homes/ydwang/EPMTL/AllModality/ST/IDH/train_features.csv'
    model_folder = r'/homes/ydwang/EPMTL/AllModality/MT/Models'
    ex_folder= r'/homes/ydwang/EPMTL/AllModality/TCIA/MT'

    Pred_MT(train_file, model_folder, ex_folder)

    