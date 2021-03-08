import numpy as np
import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from scipy import fftpack

# 音声データを読み込む
speakers = {'kirishima' : 0, 'suzutsuki' : 1, 'belevskaya' : 2}

# 特徴量を返す
def get_feat(file_name):
    a, sr = librosa.load(file_name)
    fft_wave = fftpack.rfft(a, n=sr)
    fft_freq = fftpack.rfftfreq(n=sr, d=1/sr)
    y = librosa.amplitude_to_db(fft_wave, ref=np.max)
#   plt.plot(fft_freq, y)
#   plt.show()
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

# 教師データとテストデータに分ける
train_X, test_X, train_y, test_y = train_test_split(data_X, data_y, random_state=813)
print("{} -> {}, {}".format(len(data_X), len(train_X), len(test_X)))


# 各時間に対応する成分をそれぞれ1つの特徴量として分割する
def split_feat(data_X, data_y):
    data_X2 = []
    data_y2 = []
    for X, y in zip(data_X, data_y):
        X2 = X.T
        y2 = np.array([y[0]] * X.shape[1])
        data_X2.append(X2)
        data_y2.append(y2)
    data_X2 = np.concatenate(data_X2)
    data_y2 = np.concatenate(data_y2)
    return (data_X2, data_y2)

train_X2, train_y2 = split_feat(train_X, train_y)

# clf = svm.SVC(gamma=0.0001, C=1)
clf = svm.SVC(gamma=0.0000001, C=10)
clf.fit(train_X, train_y.T[0])

def predict(X):
    result = clf.predict(X.T)
    return np.argmax(np.bincount(result))

ok_count = 0

for X, y in zip(test_X, test_y):
    actual = clf.predict(np.array([X]))[0]
    expected = y[0]
    file_name = y[1]
    ok_count += 1 if actual == expected else 0
    result = 'o' if actual == expected else 'x'
    print("{} file: {}, actual: {}, expected: {}".format(result, file_name, actual, expected))

print("{}/{}".format(ok_count, len(test_X)))

# mfccのを描画する
def mean_plot(mfccs, name):
    print(name)
    mean = np.mean(mfccs.T, axis=0)
    plt.plot(range(0,len(mean)), mean)
    plt.show()

mean_plot(get_feat("voiceset/kirishima_b01.wav"), "kirishima")
mean_plot(get_feat("voiceset/suzutsuki_b01.wav"), "suzutsuki")
mean_plot(get_feat("voiceset/belevskaya_b01.wav"), "belevskaya")
mean_plot(get_feat("sample/hi.wav"), "hi")
mean_plot(get_feat("sample/lo.wav"), "lo")
