import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
from pathlib import Path
import glob
from utils import add_landcover_labels, summarize_landcover

def load_data(data_path="../data/raw_exports"):
    """Load và gộp tất cả file CSV từ thư mục raw_exports"""
    data_path = Path(data_path)
    all_files = glob.glob(str(data_path / "*.csv"))
    
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        # Trích xuất tọa độ từ cột .geo
        try:
            df['coordinates'] = df['.geo'].apply(lambda x: eval(x)['coordinates'])
            df['longitude'] = df['coordinates'].apply(lambda x: x[0])
            df['latitude'] = df['coordinates'].apply(lambda x: x[1])
        except:
            print(f"Lỗi khi xử lý file {file}")
            continue
        dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined

def plot_landcover_distribution(df):
    """Vẽ biểu đồ phân bố các loại lớp phủ"""
    summary = summarize_landcover(df)
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=summary.reset_index(), 
               x='land_cover_name', 
               y='total_samples',
               alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.title('Phân bố các loại lớp phủ trong dataset')
    plt.ylabel('Số lượng mẫu')
    plt.xlabel('Loại lớp phủ')
    plt.tight_layout()
    plt.show()
    
    return summary

def plot_flood_ratio(df):
    """Vẽ biểu đồ tỷ lệ ngập theo loại lớp phủ"""
    summary = summarize_landcover(df)
    
    plt.figure(figsize=(15, 6))
    sns.barplot(data=summary.reset_index(), 
               x='land_cover_name', 
               y='flood_ratio',
               alpha=0.6)
    plt.xticks(rotation=45, ha='right')
    plt.title('Tỷ lệ ngập (%) theo loại lớp phủ')
    plt.ylabel('Tỷ lệ ngập (%)')
    plt.xlabel('Loại lớp phủ')
    plt.tight_layout()
    plt.show()
    
    return summary

def create_interactive_map(df, sample_size=1000):
    """Tạo bản đồ tương tác với folium"""
    # Tạo bản đồ centered ở Việt Nam với basemap đẹp hơn
    m = folium.Map(
        location=[16.0, 106.0],
        zoom_start=5,
        tiles='CartoDB positron'
    )
    
    # Thêm các basemap layers khác
    folium.TileLayer('CartoDB dark_matter').add_to(m)
    folium.TileLayer('OpenStreetMap').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Esri Satellite'
    ).add_to(m)
    
    # Color map cho các loại lớp phủ
    unique_landcover = sorted(df['land_cover'].unique())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_landcover)))
    color_map = dict(zip(unique_landcover, 
                        [f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}' 
                         for r, g, b, _ in colors]))
    
    # Tạo feature groups cho từng loại lớp phủ và ngập/không ngập
    layer_groups = {}
    for lc in unique_landcover:
        lc_name = df[df['land_cover'] == lc]['land_cover_name'].iloc[0]
        # Group cho các điểm ngập
        layer_groups[f'{lc}_flood'] = folium.FeatureGroup(
            name=f"{lc_name} (Ngập)"
        )
        # Group cho các điểm không ngập
        layer_groups[f'{lc}_non_flood'] = folium.FeatureGroup(
            name=f"{lc_name} (Không ngập)"
        )
    
    # Lấy mẫu ngẫu nhiên
    df_sample = df.sample(sample_size, random_state=42)
    
    # Thêm markers vào các layer groups
    for idx, row in df_sample.iterrows():
        lc = row['land_cover']
        is_flood = row['flood'] == 1
        color = color_map[lc]
        
        # Chọn group phù hợp
        group_key = f'{lc}_flood' if is_flood else f'{lc}_non_flood'
        
        # Tạo popup với style
        popup_html = f"""
        <div style="font-family: Arial; padding: 10px;">
            <h4 style="margin: 0 0 10px 0; color: {'#d73027' if is_flood else '#1a9850'};">
                {row['land_cover_name']}
            </h4>
            <table style="width: 100%;">
                <tr>
                    <td><b>Trạng thái:</b></td>
                    <td style="color: {'#d73027' if is_flood else '#1a9850'};">
                        {'⚠️ Ngập' if is_flood else '✓ Không ngập'}
                    </td>
                </tr>
                <tr><td><b>Độ cao:</b></td><td>{row['elevation']:.1f}m</td></tr>
                <tr><td><b>Độ dốc:</b></td><td>{row['slope']:.1f}°</td></tr>
            </table>
        </div>
        """
        
        # Tạo marker với kích thước và style khác nhau cho ngập/không ngập
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6 if is_flood else 4,
            popup=folium.Popup(popup_html, max_width=300),
            color='red' if is_flood else 'green',
            fill=True,
            fillColor=color,
            fillOpacity=0.8 if is_flood else 0.6,
            weight=2 if is_flood else 1
        ).add_to(layer_groups[group_key])
    
    # Thêm các layer groups vào map
    for group in layer_groups.values():
        group.add_to(m)
    
    # Thêm layer control với giao diện cải thiện
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    
    # Thêm title và chú thích
    title_html = '''
    <div style="position: fixed; 
                top: 10px; 
                left: 50px; 
                width: 300px; 
                z-index: 1000;
                background-color: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0,0,0,0.2);">
        <h3 style="margin: 0;">Phân bố các loại lớp phủ</h3>
        <p style="margin: 5px 0;">
            <span style="color: #d73027;">⚠️ Đỏ: Vùng ngập</span><br>
            <span style="color: #1a9850;">✓ Xanh: Vùng không ngập</span>
        </p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em; color: #666;">
            Click vào các layer ở góc phải để bật/tắt
        </p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Lưu bản đồ
    output_path = Path("../outputs/land_cover_map.html")
    output_path.parent.mkdir(exist_ok=True)
    m.save(str(output_path))
    print(f"Đã lưu bản đồ tương tác tại: {output_path}")
    
    return m

def main():
    # Load data
    print("Đang đọc dữ liệu...")
    df = load_data()
    print(f"Tổng số mẫu: {len(df):,}")
    
    # Thêm labels
    df = add_landcover_labels(df)
    
    # Vẽ biểu đồ phân bố
    print("\nĐang tạo biểu đồ phân bố...")
    summary = plot_landcover_distribution(df)
    
    # Vẽ biểu đồ tỷ lệ ngập
    print("\nĐang tạo biểu đồ tỷ lệ ngập...")
    summary = plot_flood_ratio(df)
    
    # In bảng tổng kết
    print("\nBảng tổng kết chi tiết:")
    print(summary.round(2).to_string())
    
    # Tạo bản đồ tương tác
    print("\nĐang tạo bản đồ tương tác...")
    create_interactive_map(df)
    
    print("\nHoàn tất! Kiểm tra thư mục outputs/ để xem kết quả.")

if __name__ == "__main__":
    # Set style và palette cho matplotlib
    sns.set_theme()  # Sử dụng theme mặc định của seaborn
    sns.set_palette("husl")
    
    main()