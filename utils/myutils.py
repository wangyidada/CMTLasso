import numpy as np

def MultiLabelTransform(labels, classes=2):
    labels_transform = []
    for label in labels:
        label_list =[]
        encoder_label = bin(int(label)).replace('0b','').rjust(classes, '0')
        for i in range(classes):
            label_list.append(encoder_label[i])
        labels_transform.append(label_list)
    labels_value = np.array(labels_transform, dtype=np.float16)
    return labels_value

def Transform2SingleLabel(df, i):
    if i == 0:
        df.loc[df['label'] == 1, 'label'] = 0
        df.loc[df['label'] == 2, 'label'] = 1
        df.loc[df['label'] == 3, 'label'] = 1

    elif i ==1:
        df.loc[df['label'] == 2, 'label'] = 0
        df.loc[df['label'] == 3, 'label'] = 1

    return df


if __name__ == '__main__':
    labels = [0, 1, 2, 3]
    l = MultiLabelTransform(labels, classes=2)
    print(l)
