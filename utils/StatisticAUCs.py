import os
import numpy as np
import pandas as pd
from Visualization import plot_boxes


def statistic_MLTAUC(folder, tasks):
    train_aucs, val_aucs = [], []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file, 'evaluation_metrix.csv')
        df = pd.read_csv(file_path)
        train_aucs_fold, val_aucs_fold = [], []
        for task in tasks:
            name1 = 'cv_train_task_' + task + '_auc'
            name2 = 'cv_val_task_' + task + '_auc'
            train_auc = df[df['metrix']==name1]['value'].values
            val_auc = df[df['metrix']==name2]['value'].values
            train_aucs_fold.append(train_auc)
            val_aucs_fold.append(val_auc)
        train_aucs.append(train_aucs_fold)
        val_aucs.append(val_aucs_fold)
    
    train_aucs_value = np.array(train_aucs, dtype=np.float32)[..., 0]
    val_aucs_value = np.array(val_aucs, dtype=np.float32)[..., 0]
    return train_aucs_value, val_aucs_value


def statistic_SingleAUC(folder, tasks):
    train_aucs_tasks, val_aucs_tasks = [], []
    for task in tasks:
        train_aucs_task, val_aucs_task = [], []
        task_folder = os.path.join(folder, task)
        for file in os.listdir(task_folder):
            file_path = os.path.join(task_folder, file, 'evaluation_metrix.csv')
            df = pd.read_csv(file_path)
            name1 = 'cv_train_task_' + task + '_auc'
            name2 = 'cv_val_task_' + task + '_auc'
            train_auc = df[df['metrix']==name1]['value'].values
            val_auc = df[df['metrix']==name2]['value'].values
            train_aucs_task.append(train_auc)
            val_aucs_task.append(val_auc)

        train_aucs_tasks.append(train_aucs_task)
        val_aucs_tasks.append(val_aucs_task)
    
    train_aucs_value = np.array(train_aucs_tasks, dtype=np.float32)[..., 0]
    val_aucs_value = np.array(val_aucs_tasks, dtype=np.float32)[..., 0]
    train_aucs_value = np.transpose(train_aucs_value, [1, 0])
    val_aucs_value = np.transpose(val_aucs_value, [1, 0])
    return train_aucs_value, val_aucs_value

def statistic_aucs(multi_task_folder, single_task_folder, tasks, save_folder):
    train_aucs_value, val_aucs_value = statistic_MLTAUC(multi_task_folder, tasks)
    plot_boxes(train_aucs_value, val_aucs_value, tasks, save_name=os.path.join(save_folder, 'CVMTLTaskAUCs.jpg'))
    train_aucs_value, val_aucs_value = statistic_SingleAUC(single_task_folder, tasks)
    plot_boxes(train_aucs_value, val_aucs_value, tasks, save_name=os.path.join(save_folder, 'CVSingleTaskAUCs.jpg'))



