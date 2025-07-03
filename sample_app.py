# visualization_example.py - データ可視化の例

from dataloader import HandDataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def visualize_data():
    """データを可視化する例"""
    
    # データローダーを初期化
    loader = HandDataLoader()
    
    # データを読み込み
    hand_data, emg_data = loader.load_all_data("hand_data_log.csv", "emg_data.csv")
    
    if hand_data is None or emg_data is None:
        print("Error: Could not load data files")
        return
    
    # プロット設定
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Hand Data and EMG Analysis', fontsize=16)
    
    # 1. 手の姿勢データのタイムライン
    ax1 = axes[0, 0]
    ax1.scatter(hand_data['timestamp'], hand_data['thumb_mcp_quat_0'], 
                alpha=0.6, s=1, label='Thumb MCP Quat 0')
    ax1.set_title('Hand Pose Timeline (Thumb MCP)')
    ax1.set_xlabel('Timestamp')
    ax1.set_ylabel('Quaternion Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. EMGセンサーデータ
    ax2 = axes[0, 1]
    for i in range(1, 9):  # Sensor_1 to Sensor_8
        sensor_col = f'Sensor_{i}'
        ax2.plot(emg_data['Timestamp'], emg_data[sensor_col], 
                 label=f'Sensor {i}', alpha=0.7)
    ax2.set_title('EMG Sensor Data')
    ax2.set_xlabel('Timestamp')
    ax2.set_ylabel('Sensor Value')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. 左手vs右手のデータ分布
    ax3 = axes[1, 0]
    left_hand = hand_data[hand_data['is_left'] == True]
    right_hand = hand_data[hand_data['is_left'] == False]
    
    ax3.hist(left_hand['thumb_mcp_quat_0'], bins=50, alpha=0.6, label='Left Hand', density=True)
    ax3.hist(right_hand['thumb_mcp_quat_0'], bins=50, alpha=0.6, label='Right Hand', density=True)
    ax3.set_title('Left vs Right Hand Distribution')
    ax3.set_xlabel('Thumb MCP Quat 0')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. EMGセンサーの相関マトリックス
    ax4 = axes[1, 1]
    sensor_columns = [f'Sensor_{i}' for i in range(1, 9)]
    correlation_matrix = emg_data[sensor_columns].corr()
    
    im = ax4.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    ax4.set_title('EMG Sensor Correlation Matrix')
    ax4.set_xticks(range(len(sensor_columns)))
    ax4.set_yticks(range(len(sensor_columns)))
    ax4.set_xticklabels(sensor_columns, rotation=45)
    ax4.set_yticklabels(sensor_columns)
    
    # カラーバーを追加
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Correlation')
    
    plt.tight_layout()
    plt.savefig('hand_emg_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved as 'hand_emg_analysis.png'")

def analyze_finger_movement():
    """特定の指の動きを分析"""
    
    loader = HandDataLoader()
    hand_data = loader.load_hand_data("hand_data_log.csv")
    
    if hand_data is None:
        return
    
    # 人差し指の各関節の動きを可視化
    plt.figure(figsize=(12, 8))
    
    # 人差し指の関節データ
    index_joints = ['index_mcp_quat_0', 'index_pip_quat_0', 'index_dip_quat_0']
    joint_names = ['MCP', 'PIP', 'DIP']
    
    for i, (joint, name) in enumerate(zip(index_joints, joint_names)):
        plt.subplot(2, 2, i+1)
        plt.plot(hand_data['timestamp'], hand_data[joint], label=f'Index {name}')
        plt.title(f'Index Finger {name} Joint Movement')
        plt.xlabel('Timestamp')
        plt.ylabel('Quaternion Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    # 全体の動きパターン
    plt.subplot(2, 2, 4)
    for joint, name in zip(index_joints, joint_names):
        plt.plot(hand_data['timestamp'], hand_data[joint], label=f'{name}', alpha=0.7)
    plt.title('Index Finger - All Joints')
    plt.xlabel('Timestamp')
    plt.ylabel('Quaternion Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('index_finger_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Index finger analysis saved as 'index_finger_analysis.png'")

def compare_emg_patterns():
    """EMGパターンの比較分析"""
    
    loader = HandDataLoader()
    emg_data = loader.load_emg_data("emg_data.csv")
    
    if emg_data is None:
        return
    
    # EMGデータの統計分析
    sensor_columns = [f'Sensor_{i}' for i in range(1, 9)]
    
    plt.figure(figsize=(15, 10))
    
    # 各センサーの時系列データ
    for i, sensor in enumerate(sensor_columns):
        plt.subplot(2, 4, i+1)
        plt.plot(emg_data['Timestamp'], emg_data[sensor])
        plt.title(f'{sensor}')
        plt.xlabel('Timestamp')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emg_sensors_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("EMG sensor comparison saved as 'emg_sensors_comparison.png'")

if __name__ == "__main__":
    print("=== Data Visualization Examples ===")
    
    # 基本的な可視化
    visualize_data()
    
    # 指の動きの分析
    analyze_finger_movement()
    
    # EMGパターンの比較
    compare_emg_patterns()