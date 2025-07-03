import pandas as pd
import numpy as np

class HandDataLoader:
    """手の姿勢データとEMGデータを読み込むシンプルなデータローダー"""
    
    def __init__(self):
        self.hand_data = None
        self.emg_data = None
    
    def load_hand_data(self, filepath):
        """
        手の姿勢データを読み込み
        
        Args:
            filepath (str): hand_data_log.csvのパス
            
        Returns:
            pd.DataFrame: 読み込まれた手の姿勢データ
        """
        try:
            self.hand_data = pd.read_csv(filepath)
            print(f"Hand data loaded: {len(self.hand_data)} rows, {len(self.hand_data.columns)} columns")
            return self.hand_data
        except Exception as e:
            print(f"Error loading hand data: {e}")
            return None
    
    def load_emg_data(self, filepath):
        """
        EMGデータを読み込み
        
        Args:
            filepath (str): emg_data.csvのパス
            
        Returns:
            pd.DataFrame: 読み込まれたEMGデータ
        """
        try:
            self.emg_data = pd.read_csv(filepath)
            print(f"EMG data loaded: {len(self.emg_data)} rows, {len(self.emg_data.columns)} columns")
            return self.emg_data
        except Exception as e:
            print(f"Error loading EMG data: {e}")
            return None
    
    def load_all_data(self, hand_filepath, emg_filepath):
        """
        すべてのデータを一度に読み込み
        
        Args:
            hand_filepath (str): hand_data_log.csvのパス
            emg_filepath (str): emg_data.csvのパス
            
        Returns:
            tuple: (hand_data, emg_data)
        """
        hand_data = self.load_hand_data(hand_filepath)
        emg_data = self.load_emg_data(emg_filepath)
        return hand_data, emg_data
    
    def get_basic_info(self):
        """データの基本情報を表示"""
        if self.hand_data is not None:
            print("\n=== Hand Data Info ===")
            print(f"Shape: {self.hand_data.shape}")
            print(f"Columns: {list(self.hand_data.columns)}")
            print(f"Data types:\n{self.hand_data.dtypes}")
            
        if self.emg_data is not None:
            print("\n=== EMG Data Info ===")
            print(f"Shape: {self.emg_data.shape}")
            print(f"Columns: {list(self.emg_data.columns)}")
            print(f"Data types:\n{self.emg_data.dtypes}")


# 使用例
if __name__ == "__main__":
    # データローダーを初期化
    loader = HandDataLoader()
    
    # データファイルのパスを指定
    hand_data_path = "hand_data_log.csv"
    emg_data_path = "emg_data.csv"
    
    # データを読み込み
    hand_data, emg_data = loader.load_all_data(hand_data_path, emg_data_path)
    
    # 基本情報を表示
    loader.get_basic_info()
    
    # データの最初の5行を表示
    if hand_data is not None:
        print("\n=== Hand Data Sample ===")
        print(hand_data.head())
    
    if emg_data is not None:
        print("\n=== EMG Data Sample ===")
        print(emg_data.head())