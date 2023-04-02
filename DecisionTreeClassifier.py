from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from glob import glob
from os.path import basename
import numpy as np
from micromlgen import port

def load_features(folder):
    dataset = None
    classmap = {}
    for class_idx, filename in enumerate(glob('%s/*.csv' % folder)):
        class_name = basename(filename)[:-4]
        classmap[class_idx] = class_name
        samples = np.loadtxt(filename, dtype=float, delimiter=',')
        labels = np.ones((len(samples), 1)) * class_idx
        samples = np.hstack((samples, labels))
        dataset = samples if dataset is None else np.vstack((dataset, samples))

    return dataset, classmap


features, classmap = load_features('/content')
X, y = features[:, :-1], features[:, -1]
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
classifier = DecisionTreeClassifier().fit(X_train, y_train)

# Avaliando Modelo
y_pred = classifier.predict(x_test)

# Acurácia
acuracia = (accuracy_score(y_test, y_pred) * 100)
# Precisão
precision = precision_score(y_test, y_pred) * 100
# Recall
recall = recall_score(y_test, y_pred) * 100
# F1-Score
f1 = f1_score(y_test, y_pred) * 100
# calculate AUC
auc = roc_auc_score(y_test, y_pred) * 100

# Resultados
print('')
print('MODELO - Decision Tree Classifier')
print('Acurácia: %0.2f%%' % acuracia)
print('Precisão: %0.2f%%' % precision)
print('F1-Score: %0.2f%%' % f1)
print('Recall: %0.2f%%' % recall)
print('AUC: %.2f%%' % auc)

c_code = port(classifier, classmap=classmap)
print(c_code)
