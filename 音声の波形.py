import librosa
import librosa.display
import matplotlib.pyplot as plt

# 再生した音声の波形を描画する

a, sr = librosa.load('voiceset/kirishima_b01.wav')  #サンプルボイスをいれる
librosa.display.waveplot(a, sr)

# 波形を表すデータ: 振幅の数値の列
print(a)
print(len(a))

# サンプリングレート -> 1秒間のフレームの数
print(sr)

# 6秒の音声なので、次がlen(a)と一致するはず
print(sr * 6)

# 高音
a, sr = librosa.load('sample/hi.wav')　　#高音のサンプルボイスをいれる
librosa.display.waveplot(a, sr)
plt.show()

# 低音
a, sr = librosa.load('sample/lo.wav')　　#低音のサンプルボイスをいれる
librosa.display.waveplot(a, sr)
plt.show()

# 高音
a, sr = librosa.load('sample/hi.wav')
librosa.display.waveplot(a[0:100], sr)
plt.show()

# 低音
a, sr = librosa.load('sample/lo.wav')
librosa.display.waveplot(a[0:100], sr)
plt.show()
