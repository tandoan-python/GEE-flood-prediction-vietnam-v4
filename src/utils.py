import pandas as pd

# Mapping cho land_cover
LAND_COVER_MAPPING = {
    0: "Không có dữ liệu",
    10: "Rừng (>5m)",
    20: "Cây bụi (0.5-5m)",
    30: "Đồng cỏ",
    40: "Đất nông nghiệp",
    50: "Khu dân cư/Đô thị",
    60: "Đất trống/Cây thưa",
    70: "Tuyết/Băng",
    80: "Mặt nước thường xuyên",
    90: "Đất ngập nước",
    95: "Rừng ngập mặn",
    100: "Rêu/Địa y"
}

# Mapping cho mức độ rủi ro ngập
FLOOD_RISK_MAPPING = {
    0: "không xác định",
    10: "thấp",
    20: "trung bình",
    30: "trung bình",
    40: "cao",
    50: "cao",
    60: "trung bình",
    70: "thấp",
    80: "đã là nước",
    90: "cao",
    95: "đặc biệt",
    100: "thấp"
}

def add_landcover_labels(df):
    """Thêm cột tên lớp phủ và mức độ rủi ro ngập vào DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame có cột 'land_cover'
    
    Returns:
        pandas.DataFrame: DataFrame với thêm 2 cột mới:
            - land_cover_name: tên lớp phủ tiếng Việt
            - flood_risk: mức độ rủi ro ngập
    """
    if 'land_cover' not in df.columns:
        raise ValueError("DataFrame phải có cột 'land_cover'")
    
    # Thêm tên lớp phủ tiếng Việt
    df['land_cover_name'] = df['land_cover'].map(LAND_COVER_MAPPING)
    
    # Thêm mức độ rủi ro ngập
    df['flood_risk'] = df['land_cover'].map(FLOOD_RISK_MAPPING)
    
    return df

def summarize_landcover(df):
    """Tạo bảng tổng kết về phân bố lớp phủ và tỷ lệ ngập.
    
    Args:
        df (pandas.DataFrame): DataFrame có các cột: 'land_cover', 'flood'
    
    Returns:
        pandas.DataFrame: Bảng tổng kết với các cột:
            - land_cover_name: tên lớp phủ
            - total_samples: tổng số mẫu
            - flood_samples: số mẫu bị ngập
            - flood_ratio: tỷ lệ ngập (%)
    """
    if not {'land_cover', 'flood'}.issubset(df.columns):
        raise ValueError("DataFrame phải có cả 2 cột 'land_cover' và 'flood'")
    
    # Thêm tên lớp phủ
    df = add_landcover_labels(df)
    
    # Tạo bảng tổng kết
    summary = df.groupby('land_cover_name').agg({
        'land_cover': 'count',
        'flood': 'sum'
    }).rename(columns={
        'land_cover': 'total_samples',
        'flood': 'flood_samples'
    })
    
    # Tính tỷ lệ ngập
    summary['flood_ratio'] = (summary['flood_samples'] / summary['total_samples'] * 100).round(2)
    
    return summary.sort_values('total_samples', ascending=False)