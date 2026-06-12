import numpy as np
import struct
from osgeo import gdal, osr

# 显式声明不使用异常，消除 FutureWarning 警告
gdal.DontUseExceptions()

def create_mock_chongqing_dem(filename="cq_dem_meters.tif"):
    print("正在本地生成仿真重庆平行岭谷 DEM 数据（安全流模式）...")
    
    width = 500
    height = 500
    
    # 1. 依然使用数学公式模拟重庆条状岭谷
    x = np.linspace(0, 4 * np.pi, width)
    y = np.linspace(0, 2 * np.pi, height)
    X, Y = np.meshgrid(x, y)
    
    np.random.seed(42)
    noise = np.random.normal(0, 8, (height, width))
    elevation = 250 + np.abs(np.sin(X)) * 600 + noise
    
    # 强制转换为标准的 32位浮点数 矩阵
    elevation = elevation.astype(np.float32)

    # 2. 创建真正的 GeoTIFF 文件
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(filename, width, height, 1, gdal.GDT_Float32)
    
    # 3. 设置地理仿射变换参数与坐标系 (UTM 49N)
    pixel_size = 30.0
    top_left_x = 350000.0  
    top_left_y = 3200000.0
    dataset.SetGeoTransform([top_left_x, pixel_size, 0, top_left_y, 0, -pixel_size])
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32649)
    dataset.SetProjection(srs.ExportToWkt())
    
    # 4. 【核心改动】：绕过 WriteArray，改用底层的 WriteRaster
    # 将 numpy 矩阵直接打碎转化为底层的二进制 C++ float 字节流
    bin_data = elevation.tobytes()
    
    band = dataset.GetRasterBand(1)
    # 参数含义: (起点X, 起点Y, 写入宽度, 写入高度, 二进制流)
    band.WriteRaster(0, 0, width, height, bin_data)
    
    band.FlushCache()
    dataset = None # 显式释放对象，确保数据完整写入磁盘
    
    print(f" 仿真数据生成成功！已保存为: {filename}")

if __name__ == "__main__":
    create_mock_chongqing_dem()