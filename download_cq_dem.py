import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def download_chongqing_dem_fallback(output_filename="chongqing_srtm30.tif"):
    print(" 正在尝试通过备用通道获取重庆 DEM 数据...")
    
    # 依然是重庆主城及周边的核心范围
    south, north, west, east = 29.00, 30.50, 106.00, 107.50
    
    # 换用 OpenTopography 的另一个稳定基准 URL，或者你可以直接测试这个请求
    url = f"https://portal.opentopography.org/api/globaldem?demtype=SRTMGL1&south={south}&north={north}&west={west}&east={east}&outputFormat=GTiff"
    
    # 建立强大的重试机制，防止网络抖动
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        print("正在连接地理数据服务器，这可能需要1-2分钟...")
        response = session.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(output_filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f" 成功！数据已保存在: {os.path.abspath(output_filename)}")
            return True
        else:
            print(f"服务器回应错误码: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"网络连接依然受阻，错误原因: {e}")
        return False

if __name__ == "__main__":
    download_chongqing_dem_fallback()