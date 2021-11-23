import pandas as pd
import random
import numpy as np

# Load Train's and Valid's Labels.csv
label_data = pd.read_csv('data_info.csv')
# print(label_data)

# Shuffle and Seperate Train's and Valid's Labels
# Negative
data = label_data['Negative']
Negative_data = list(label_data['FileID'][data==1])
# Typical
data = label_data['Typical']
Typical_data = list(label_data['FileID'][data==1])
# Atypical
data = label_data['Atypical']
Atypical_data = list(label_data['FileID'][data==1])

print(len(Negative_data))
print(len(Typical_data))
print(len(Atypical_data))

random.seed(2021)
random.shuffle(Negative_data)
# print(Negative_data)

random.shuffle(Typical_data)
# print(Typical_data)

random.shuffle(Atypical_data)
# print(Atypical_data)

train_label_list = []
for i in range(3):
    for j in range(350):
        train_label_list.append(i)

valid_label_list = []
for i in range(3):
    for j in range(50):
        valid_label_list.append(i)

random.seed(2021)
random.shuffle(train_label_list)
random.shuffle(valid_label_list)

n_index = 0
t_index = 0
a_index = 0
train_file_list = []
for i in train_label_list:
    if i == 0:
        train_file_list.append(Negative_data[n_index])
        n_index += 1
    elif i == 1:
        train_file_list.append(Typical_data[t_index])
        t_index += 1
    else:
        train_file_list.append(Atypical_data[a_index])
        a_index += 1

valid_file_list = []
for i in valid_label_list:
    if i == 0:
        valid_file_list.append(Negative_data[n_index])
        n_index += 1
    elif i == 1:
        valid_file_list.append(Typical_data[t_index])
        t_index += 1
    else:
        valid_file_list.append(Atypical_data[a_index])
        a_index += 1

print(n_index)
print(t_index)
print(a_index)


# Produce Train's and Valid's Labels.txt
with open('./data/train_labels.txt', 'w') as txtfile:
    for i in range(len(train_file_list)):
        txtfile.write('{}.jpg {}\n'.format(train_file_list[i], train_label_list[i]))

with open('./data/valid_labels.txt', 'w') as txtfile:
    for i in range(len(valid_file_list)):
        txtfile.write('{}.jpg {}\n'.format(valid_file_list[i], valid_label_list[i]))

# Load Fake Test's Labels.csv
label_data = pd.read_csv('sample_submission.csv')
# print(label_data)

# Produce Fake Test's Labels.txt
test_file_list = list(label_data['FileID'])
test_label_list = []
for i in range(len(test_file_list)):
    if label_data['Type'][i] == 'Negative':
        test_label_list.append(0)
    elif label_data['Type'] == 'Typical':
        test_label_list.append(1)
    else:
        test_label_list.append(2)

with open('./data/sample_labels.txt', 'w') as txtfile:
    for i in range(len(test_file_list)):
        txtfile.write('{}.jpg {}\n'.format(test_file_list[i], test_label_list[i]))