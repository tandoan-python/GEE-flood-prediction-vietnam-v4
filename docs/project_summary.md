# Dự án Dự báo Lũ lụt sử dụng Google Earth Engine

## Tổng quan
Dự án này xây dựng mô hình dự báo lũ lụt cho Việt Nam sử dụng dữ liệu vệ tinh từ Google Earth Engine (GEE) và các kỹ thuật học máy.

## Cấu trúc dự án

```
├── app/                    # Ứng dụng web (API và Dashboard)
├── data/                   # Thư mục dữ liệu
│   ├── processed/         # Dữ liệu đã xử lý
│   └── raw_exports/       # Dữ liệu thô từ GEE
├── models/                # Mô hình đã huấn luyện
├── src/                   # Mã nguồn
└── outputs/               # Kết quả đánh giá
```

## Dữ liệu huấn luyện
Dự án sử dụng 15 sự kiện lũ lụt lịch sử:
- 10 sự kiện cho tập huấn luyện (Training)
- 3 sự kiện cho tập kiểm định (Validation)
- 2 sự kiện cho tập kiểm thử (Testing)

## Các đặc trưng (Features)

### Đặc trưng tĩnh
1. **elevation**: Cao độ (DEM)
2. **slope**: Độ dốc
3. **aspect**: Hướng sườn
4. **land_cover**: Lớp phủ mặt đất (ESA WorldCover)
5. **soil_type**: Loại đất

#### Các flags từ land_cover:
- **is_flood_prone**: Khu vực dễ ngập (nông nghiệp, đô thị, đất ngập nước)
- **is_permanent_water**: Vùng nước thường xuyên
- **is_urban**: Khu vực đô thị
- **is_agriculture**: Đất nông nghiệp

### Đặc trưng động
1. **precip_total**: Lượng mưa tích lũy trong sự kiện
2. **precip_14_day**: Lượng mưa 14 ngày trước
3. **precip_7_day**: Lượng mưa 7 ngày trước
4. **precip_3_day**: Lượng mưa 3 ngày trước
5. **soil_moisture**: Độ ẩm đất

## Quy trình xử lý dữ liệu

1. **Phân tích Sentinel-1 (SAR)**:
   - Baseline: Dữ liệu 3 tháng trước lũ
   - During: Dữ liệu trong thời gian lũ
   - Phát hiện thay đổi để xác định vùng ngập

2. **Lấy mẫu**:
   - Scale: 90m (để giảm tải GEE)
   - 3000 điểm cho mỗi lớp (ngập/không ngập)
   - Chỉ lấy mẫu ở vùng có độ dốc < 20°

## Các ngưỡng quan trọng

1. **Phát hiện ngập (flood=1)**:
   - Sentinel-1 diff > 2.5 dB
   - Slope < 20°

2. **Xác định không ngập (flood=0)**:
   - Sentinel-1 diff < 1.5 dB
   - Slope < 20°

## Quy trình xuất dữ liệu

1. Khởi tạo tasks trong GEE
2. Chạy các tasks trong GEE Code Editor
3. Tải dữ liệu từ Google Drive (thư mục GEE_Flood_Exports)
4. Di chuyển files CSV vào thư mục data/raw_exports/
5. Chạy script combine_data.py để tổng hợp

## Tối ưu hóa
- Sử dụng high-volume endpoint của GEE
- tileScale=16 cho các phép toán nặng
- Sử dụng scale=90m để giảm tải GEE
- Xử lý từng sự kiện với delay để tránh quá tải

## Hạn chế và lưu ý
1. Cần xác thực GEE trước khi chạy
2. Có thể thiếu dữ liệu Sentinel-1 cho một số sự kiện
3. Cần kiểm tra kỹ các task trong GEE Code Editor
4. Thời gian xử lý có thể kéo dài do khối lượng dữ liệu lớn