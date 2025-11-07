import pandas as pd
from pathlib import Path

def inspect_data():
    # Load một file CSV để kiểm tra
    data_path = Path("../data/raw_exports")
    files = list(data_path.glob("*.csv"))
    
    if not files:
        print("Không tìm thấy file CSV nào trong thư mục data/raw_exports/")
        return
        
    # Load file đầu tiên
    first_file = files[0]
    print(f"Đang đọc file: {first_file}")
    
    df = pd.read_csv(first_file)
    
    print("\nCác cột trong DataFrame:")
    print(df.columns.tolist())
    
    print("\nMẫu dữ liệu đầu tiên:")
    print(df.iloc[0])
    
    # Kiểm tra xem có cột chứa thông tin địa lý không
    geo_cols = [col for col in df.columns if 'geo' in col.lower()]
    if geo_cols:
        print("\nCác cột chứa thông tin địa lý:")
        print(geo_cols)
        print("\nMẫu dữ liệu địa lý:")
        print(df[geo_cols].iloc[0])

if __name__ == "__main__":
    inspect_data()