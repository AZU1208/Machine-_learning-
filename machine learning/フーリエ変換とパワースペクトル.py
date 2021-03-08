import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import fftpack

# 音声データを読み込む
dir_name = 'voiceset'
for file_name in os.listdir(path=dir_name):
    print("read: {}".format(file_name))
    a, sr = librosa.load(os.path.join(dir_name, file_name))
    print(a.shape)
    librosa.display.waveplot(a, sr)
    plt.show()

# 音声データを読み込む
speakers = {'kirishima' : 0, 'suzutsuki' : 1, 'belevskaya' : 2}

# 特徴量を返す
def get_feat(file_name):
    a, sr = librosa.load(file_name)
    fft_wave = fftpack.rfft(a, n=sr)
    fft_freq = fftpack.rfftfreq(n=sr, d=1/sr)
    y = librosa.amplitude_to_db(fft_wave, ref=np.max)
#     plt.plot(fft_freq, y)
#     plt.show()
    return y

# 特徴量と分類のラベル済みのラベルの組を返す
def get_data(dir_name):
    data_X = []
    data_y = []
    for file_name in sorted(os.listdir(path=dir_name)):
        print("read: {}".format(file_name))
        speaker = file_name[0:file_name.index('_')]
        data_X.append(get_feat(os.path.join(dir_name, file_name)))
        data_y.append((speakers[speaker], file_name))
    return (np.array(data_X), np.array(data_y))

data_X, data_y = get_data('voiceset')
# get_feat('sample/hi.wav')
# get_feat('sample/lo.wav')

print("===== data_X =====")
print(data_X.shape)
print(data_X)
print("===== data_y =====")
print(data_y.shape)
print(data_y)

# 教師データとテストデータに分ける
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, random_state=11813)
print("{} -> {}, {}".format(len(data_X), len(train_X), len(test_X)))

# clf = svm.SVC(gamma =0.0001, C=1)
clf = svm.SVC(gamma=0.0000001, C=10)
clf.fit(train_X, train_y.T[0])

clf.predict(np.array([test_X[0]]))

ok_count = 0

for X, y in zip(test_X, test_y):
    actual = clf.predict(np.array([X]))[0]
    expected = y[0]
    file_name = y[1]
    ok_count += 1 if actual == expected else 0
    result = 'o' if actual == expected else 'x'
    print("{} file: {}, actual: {}, expected: {}".format(result, file_name, actual, expected))

print("{}/{}".format(ok_count, len(test_X)))
