import streamlit as st
from streamlit_folium import st_folium
import folium
import requests
import pandas as pd
import altair as alt
import datetime

# =============================================================================
# CAU HINH TRANG
# =============================================================================
st.set_page_config(
    page_title="Hệ thống Dự báo Ngập lụt",
    page_icon="🌊",
    layout="wide"
)

# Dia chi API backend
API_URL = "http://127.0.0.1:8000"

# =============================================================================
# KHOI TAO STATE
# =============================================================================
if 'map_center' not in st.session_state:
    st.session_state.map_center = [16.047079, 108.206230]  # Da Nang
if 'last_clicked' not in st.session_state:
    st.session_state.last_clicked = None
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'forecast_data' not in st.session_state:
    st.session_state.forecast_data = None
if 'error_message' not in st.session_state:
    st.session_state.error_message = None

# =============================================================================
# FUNCTIONS
# =============================================================================
def format_probability(prob):
    """Format xác suất ngập thành văn bản và màu sắc"""
    prob_percent = prob * 100
    if prob_percent > 70:
        return "🔴 Cao", "red", f"{prob_percent:.1f}%"
    elif prob_percent > 30:
        return "🟡 Trung bình", "orange", f"{prob_percent:.1f}%"
    else:
        return "🟢 Thấp", "green", f"{prob_percent:.1f}%"

# =============================================================================
# HEADER
# =============================================================================
st.title("🌊 Hệ thống Dự báo Ngập lụt")
st.caption("Dự báo ngập lụt dựa trên mô hình XGBoost và dữ liệu vệ tinh:")
st.caption("- Địa hình: SRTM")
st.caption("- Lớp phủ: ESA WorldCover")
st.caption("- Lượng mưa quá khứ: GPM IMERG")
st.caption("- Dự báo mưa: NOAA GFS (tạm thời không sử dụng; dashboard sẽ hiển thị xác nhận nếu không có dự báo mưa)")
st.caption("- Độ ẩm đất: NASA SMAP")

# =============================================================================
# LAYOUT
# =============================================================================
col1, col2 = st.columns([2, 1])

# =============================================================================
# COT 1: BAN DO
# =============================================================================
with col1:
    st.subheader("Bản đồ Tương tác")
    st.info("👆 Nhấp vào một vị trí trên bản đồ để xem dự báo")
    
    # Tao ban do
    m = folium.Map(location=st.session_state.map_center, zoom_start=10)
    
    # Them marker
    if st.session_state.last_clicked:
        folium.Marker(
            [st.session_state.last_clicked['lat'], 
             st.session_state.last_clicked['lng']],
            popup="Vị trí đã chọn",
            icon=folium.Icon(color="red"),
        ).add_to(m)

    # Hien thi ban do
    map_data = st_folium(m, width='100%', height=500)
    
    # Xu ly khi click
    if map_data and map_data['last_clicked']:
        clicked_point = map_data['last_clicked']
        if clicked_point != st.session_state.last_clicked:
            st.session_state.last_clicked = clicked_point
            point_data = {
                "lat": clicked_point['lat'], 
                "lon": clicked_point['lng']
            }
            
            with st.spinner("⏳ Đang lấy dữ liệu và dự đoán..."):
                try:
                    # Gọi API dự đoán hiện tại
                    predict_response = requests.post(
                        f"{API_URL}/predict", 
                        json=point_data
                    )
                    predict_response.raise_for_status()
                    predict_data = predict_response.json()
                    st.session_state.current_prediction = predict_data
                    
                    # Gọi API dự báo
                    forecast_response = requests.post(
                        f"{API_URL}/forecast", 
                        json=point_data
                    )
                    forecast_response.raise_for_status()
                    st.session_state.forecast_data = forecast_response.json()
                    
                    # Debug response và status code
                    with st.expander("🔍 Debug: API Response"):
                        st.write(f"Status Code: {forecast_response.status_code}")
                        st.write("Response Headers:")
                        st.json(dict(forecast_response.headers))
                        st.write("Response Data:")
                        st.json(st.session_state.forecast_data)
                    
                except requests.exceptions.RequestException as e:
                    try:
                        detail = e.response.json().get('detail', str(e))
                        st.session_state.error_message = f"Lỗi API: {detail}"
                    except:
                        st.session_state.error_message = f"Lỗi kết nối: {str(e)}"

# =============================================================================
# COT 2: KET QUA DU BAO
# =============================================================================
with col2:
    if st.session_state.error_message:
        st.error(st.session_state.error_message)
        
    elif st.session_state.last_clicked is None:
        st.info("👈 Vui lòng chọn một điểm trên bản đồ")
        
    else:
        # Tạo tabs
        tab1, tab2 = st.tabs([
            "📊 Dự báo hiện tại",
            "📅 Dự báo 7 ngày tới"
        ])
        
        # Tab 1: Dự báo hiện tại
        with tab1:
            if st.session_state.current_prediction:
                prob = st.session_state.current_prediction['probability']
                features = st.session_state.current_prediction['features']
                
                # Hiển thị xác suất
                level, color, prob_text = format_probability(prob)
                st.markdown(f"### Nguy cơ ngập: {level}")
                st.markdown(f"""
                <div style="padding: 20px; background-color: {color}20; 
                border-radius: 10px; text-align: center;">
                    <h1 style="color: {color}; margin: 0;">{prob_text}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Chi tiết đặc điểm
                with st.expander("📍 Đặc điểm địa hình và khu vực"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Địa hình:**")
                        st.write(f"- 🏔️ Độ cao: {features['elevation']:.1f}m")
                        st.write(f"- 📐 Độ dốc: {features['slope']:.1f}°")
                    with col2:
                        st.write("**Lớp phủ:**")
                        flags = {
                            'is_flood_prone': '🌊 Vùng dễ ngập',
                            'is_permanent_water': '💧 Vùng nước',
                            'is_urban': '🏘️ Khu dân cư',
                            'is_agriculture': '🌾 Đất nông nghiệp'
                        }
                        for flag, text in flags.items():
                            if features[flag] > 0:
                                st.info(text)
                
                # Lượng mưa lịch sử
                with st.expander("🌧️ Lượng mưa tích lũy"):
                    st.write(f"- 3 ngày: {features['precip_3_day']:.1f}mm")
                    st.write(f"- 7 ngày: {features['precip_7_day']:.1f}mm")
                    st.write(f"- 14 ngày: {features['precip_14_day']:.1f}mm")
        
        # Tab 2: Dự báo 7 ngày
        with tab2:
            if st.session_state.forecast_data:
                # DEBUG: Hiển thị raw data và kiểm tra cấu trúc
                with st.expander("Debug: Raw Forecast Data"):
                    st.json(st.session_state.forecast_data)
                    st.write("---")
                    st.write("Kiểm tra cấu trúc dữ liệu:")
                    st.write(f"- Có key 'forecast'?: {'forecast' in st.session_state.forecast_data}")
                    if 'forecast' in st.session_state.forecast_data:
                        st.write(f"- Số ngày dự báo: {len(st.session_state.forecast_data['forecast'])}")
                        st.write("- Cấu trúc ngày đầu tiên:")
                        if len(st.session_state.forecast_data['forecast']) > 0:
                            st.write(st.session_state.forecast_data['forecast'][0])
                
                forecast = st.session_state.forecast_data.get('forecast', [])
                
                # If the API indicates no rainfall forecast is used, show a clear note
                rain_used = st.session_state.forecast_data.get('rain_forecast_used', True)
                if not rain_used:
                    st.warning("Lưu ý: API đang không sử dụng dự báo mưa. Giá trị lượng mưa trong danh sách có thể là null/None.")

                # Hiển thị từng ngày (an toàn khi precipitation có thể là None)
                for day in forecast:
                    date = pd.to_datetime(day['date']).strftime("%d/%m/%Y")
                    prob = day['flood_probability']
                    rain = day.get('precipitation_mm_24hr', None)
                    
                    # Format rain safely
                    if rain is None:
                        rain_text = "—"
                    else:
                        try:
                            rain_text = f"{float(rain):.1f}mm"
                        except Exception:
                            rain_text = str(rain)

                    level, color, prob_text = format_probability(prob)

                    st.markdown(f"""
                    <div style="padding: 10px; margin: 5px 0; border: 1px solid #ddd; 
                    border-radius: 5px; background-color: {color}10">
                        <div style="display: flex; justify-content: space-between; 
                        align-items: center;">
                            <div>
                                <strong>{date}</strong>
                            </div>
                            <div>
                                🌧️ {rain_text} &nbsp;|&nbsp; 
                                <span style="color: {color}">{prob_text}</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Vẽ biểu đồ chi tiết
                if 'detail' in st.session_state.forecast_data:
                    detail = st.session_state.forecast_data['detail']
                    # Only render 3h precipitation chart if forecast_3h exists
                    if 'forecast_3h' in detail and detail.get('forecast_3h'):
                        st.write("### 📊 Dự báo mưa chi tiết")
                        df = pd.DataFrame(detail['forecast_3h'])
                        df['time'] = pd.to_datetime(df['time'])
                        
                        chart = alt.Chart(df).mark_bar().encode(
                            x=alt.X('time:T', 
                                  title='Thời gian',
                                  axis=alt.Axis(format="%d/%m %H:00", 
                                              labelAngle=-45)),
                            y=alt.Y('precipitation_mm_3hr:Q', 
                                  title='Lượng mưa (mm/3h)'),
                            tooltip=[
                                alt.Tooltip('time:T', 
                                          title='Thời gian', 
                                          format="%Y-%m-%d %H:00"), 
                                alt.Tooltip('precipitation_mm_3hr:Q', 
                                          title='Lượng mưa (mm/3h)', 
                                          format=".1f")
                            ]
                        ).properties(
                            title="Lượng mưa dự báo (3 giờ một lần)",
                            height=300
                        ).interactive()
                        
                        st.altair_chart(chart, use_container_width=True)