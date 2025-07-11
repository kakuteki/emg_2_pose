import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks

# パラメータ設定 (必要に応じて調整してください)
SAMPLING_RATE = 200  # サンプリングレート (Hz)
FILENAME = 'emg_data.csv'  # ファイル名
EMG_COLUMN = 'emg'  # EMGデータの列名

# データの読み込み
try:
    df = pd.read_csv(FILENAME)
    print("データの先頭5行を表示:")
    print(df.head())
    
    # データの確認
    if EMG_COLUMN not in df.columns:
        print(f"\n使用可能な列: {df.columns.tolist()}")
        EMG_COLUMN = input("EMGデータの列名を入力してください: ")
    
    emg_data = df[EMG_COLUMN].values
    n = len(emg_data)
    
    # FFTの計算
    yf = fft(emg_data)
    xf = fftfreq(n, 1/SAMPLING_RATE)[:n//2]
    
    # パワースペクトル密度 (PSD) の計算
    psd = 2.0/n * np.abs(yf[0:n//2])
    
    # 周波数帯域の定義 (Hz)
    frequency_bands = [
        (0, 20, 'Delta-Theta (0-20Hz)'),
        (20, 50, 'Low Beta (20-50Hz)'),
        (50, 100, 'High Beta (50-100Hz)'),
        (100, 200, 'Gamma (100-200Hz)'),
        (200, 500, 'High Gamma (200-500Hz)')
    ]
    
    # 各周波数帯域のピーク値を計算
    print("\n各周波数帯域のピーク値:")
    print("-" * 50)
    for low, high, band_name in frequency_bands:
        # 周波数帯域内のインデックスを取得
        band_mask = (xf >= low) & (xf <= high)
        if not any(band_mask):
            print(f"{band_name}: データがありません")
            continue
            
        # 帯域内のPSDを取得
        band_psd = psd[band_mask]
        band_freq = xf[band_mask]
        
        # ピークを検出
        peaks, _ = find_peaks(band_psd, height=0)
        
        if len(peaks) > 0:
            # 最大ピークを取得
            max_peak_idx = peaks[np.argmax(band_psd[peaks])]
            peak_freq = band_freq[max_peak_idx]
            peak_value = band_psd[max_peak_idx]
            print(f"{band_name}: ピーク周波数 = {peak_freq:.2f} Hz, ピーク値 = {peak_value:.2e} (dB = {20*np.log10(peak_value):.2f} dB)")
        else:
            # ピークが見つからない場合は最大値を使用
            max_idx = np.argmax(band_psd)
            peak_freq = band_freq[max_idx]
            peak_value = band_psd[max_idx]
            print(f"{band_name}: ピーク検出なし, 最大値 = {peak_value:.2e} (dB = {20*np.log10(peak_value):.2f} dB) @ {peak_freq:.2f} Hz")
    
    print("-" * 50)
    
    # プロットの作成
    plt.figure(figsize=(12, 10))
    
    # 時間領域のプロット
    plt.subplot(2, 1, 1)
    time = np.arange(0, n) / SAMPLING_RATE
    plt.plot(time, emg_data)
    plt.title('EMG Signal (Time Domain)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    # 周波数領域のプロット
    plt.subplot(2, 1, 2)
    line, = plt.plot(xf, 20 * np.log10(psd))  # dBスケールで表示
    
    # 周波数帯域を色分けして表示
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'plum']
    for i, (low, high, band_name) in enumerate(frequency_bands):
        if i < len(colors):
            plt.axvspan(low, high, alpha=0.2, color=colors[i], label=band_name)
    
    # 各帯域のピークに注釈を追加
    for low, high, band_name in frequency_bands:
        band_mask = (xf >= low) & (xf <= high)
        if not any(band_mask):
            continue
        
        band_psd = psd[band_mask]
        band_freq = xf[band_mask]
        max_idx = np.argmax(band_psd)
        peak_freq = band_freq[max_idx]
        peak_value = band_psd[max_idx]
        
        # ピークに注釈を追加
        if peak_value > 0:
            plt.annotate(f'{peak_freq:.1f}Hz',
                        xy=(peak_freq, 20 * np.log10(peak_value)),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.8))
    
    plt.title('EMG Signal (Frequency Domain) with Frequency Bands')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB)')
    plt.xlim([0, SAMPLING_RATE/2])  # ナイキスト周波数まで表示
    plt.grid(True)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 画像として保存
    plt.savefig('emg_fft_analysis.png', dpi=300)
    print("\nFFT解析結果を 'emg_fft_analysis.png' として保存しました。")
    
    # プロットを表示
    plt.show()
    
except FileNotFoundError:
    print(f"エラー: ファイル '{FILENAME}' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {str(e)}")
    print("\n使用可能な列:")
    print(df.columns.tolist())
