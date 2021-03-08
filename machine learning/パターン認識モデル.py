import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import svm

# 座標を作る
N = 15
random.seed(11813)
train_X = np.array([[random.randint(0, 100), random.randint(0, 100)] for i in range(N)])

train_X

# 描画する
for i in range(len(train_X)):
    plt.plot(train_X[i][0], train_X[i][1], 'o', color='blue')
    plt.annotate(i, (train_X[i][0], train_X[i][1]), size=20)

# 手で分類する
train_y = np.array([0 for i in range(N)])
train_y[2] = train_y[3] = train_y[1] = train_y[4] = train_y[5] = train_y[6] = train_y[11] = 1

# 分類したものを描画する
colors = ['blue', 'red']
for i in range(len(train_X)):
    plt.plot(train_X[i][0], train_X[i][1], 'o', color=colors[train_y[i]])
    plt.annotate(i, (train_X[i][0], train_X[i][1]), size=20)

# 分類したものを描画する
colors = ['blue', 'red']
for i in range(len(train_X)):
    plt.plot(train_X[i][0], train_X[i][1], 'o', color=colors[train_y[i]])
    plt.annotate(i, (train_X[i][0], train_X[i][1]), size=20)

# テストデータ。0か1かどっちだろう？
test_X = np.array([[30, 60]])

# test_Xを描画する
plt.plot(test_X[0][0], test_X[0][1], 'x', color='black')
plt.annotate('test', (test_X[0][0], test_X[0][1]), size=20)

# 分類したものを描画する
colors = ['blue', 'red']
for i in range(len(train_X)):
    plt.plot(train_X[i][0], train_X[i][1], 'o', color=colors[train_y[i]])
    plt.annotate(i, (train_X[i][0], train_X[i][1]), size=20)

# テストデータ。0か1かどっちだろう？
test_X = np.array([[30, 60]])

# test_Xを描画する
plt.plot(test_X[0][0], test_X[0][1], 'x', color='black')
plt.annotate('test', (test_X[0][0], test_X[0][1]), size=20)

# 学習する
clf = svm.SVC(gamma=0.0001, C=1)
clf.fit(train_X, train_y)

# 分類する
test_y = clf.predict(test_X)
test_y

# 分類したものを描画する
colors = ['blue', 'red']
for i in range(len(train_X)):
    plt.plot(train_X[i][0], train_X[i][1], 'o', color=colors[train_y[i]])
    plt.annotate(i, (train_X[i][0], train_X[i][1]), size=20)

# テストデータ。0か1かどっちだろう？
# test_X = np.array([[30, 60]])
test_X = np.array([[30, 60], [90, 90], [50, 50], [60, 40]])

# test_Xを描画する
# plt.plot(test_X[0][0], test_X[0][1], 'x', color='black')
# plt.annotate('test', (test_X[0][0], test_X[0][1]), size=20)

# 学習する
clf = svm.SVC(gamma=0.01, C=1)
clf.fit(train_X, train_y)

# 分類する
test_y = clf.predict(test_X)
# test_y

# text_yを描画する
for i in range(len(test_X)):
    plt.plot(test_X[i][0], test_X[i][1], 'x', color=colors[test_y[i]])
    plt.annotate('test', (test_X[i][0], test_X[i][1]), size=20)

# 決定境界を描画する
x = np.linspace(0, 100, 30)
y = np.linspace(0, 100, 30)
yy, xx = np.meshgrid(y, x)
xy = np.vstack([xx.ravel(), yy.ravel()]).T
P = clf.decision_function(xy).reshape(xx.shape)
plt.contour(xx, yy, P, colors='k',
                      levels=[0], alpha=0.5,
                      linestyles=['-'])
