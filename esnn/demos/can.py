import numpy as np
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from esnn.esnn import ESNN
from esnn.encoder import Encoder

t_start = time()
X = np.load('../../dataset/mix_x.npy')
y = np.load('../../dataset/mix_y.npy')
k = 100000
index = np.random.choice(X.shape[0], k, replace=False)
X_sub = X[index]
y_sub = y[index]
normal = np.count_nonzero(y_sub == 0)
dos = np.count_nonzero(y_sub == 1)
fuzzy = np.count_nonzero(y_sub == 2)
gear = np.count_nonzero(y_sub == 3)
rpm = np.count_nonzero(y_sub == 4)
print(normal, dos, fuzzy, gear, rpm)
print(y_sub.shape[0] == normal + dos + fuzzy + gear + rpm)
X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.2)

encoder = Encoder(20, 1.5, 0, 15)
esnn = ESNN(encoder, m=0.9, c=0.7, s=0.6)
esnn.train(X_train, y_train)
y_pred = esnn.test(X_test)

acc = accuracy_score(y_test, y_pred)
t_end = time()

print(f"Neuron Count: {len(esnn.all_neurons)}")
print(f"Accuracy: {acc}")
print("Train and test finished in ", (t_end - t_start)/60, "minutes.")
