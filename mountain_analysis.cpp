#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <iomanip>
#include "gdal_priv.h"
#include "cpl_conv.h" 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 结构体：存储分析结果（含直方图分布）
struct MountainMetrics {
    double min_elevation;
    double max_elevation;
    double relative_height;
    double max_slope;
    double avg_slope;
    std::map<int, long long> slope_distribution; 
    long long total_valid_points;
};

// 核心测算函数
MountainMetrics AnalyzeMountain(const std::string& dem_path, int col_offset, int row_offset, int width, int height) {
    GDALAllRegister();
    
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(dem_path.c_str(), GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "打开 DEM 文件失败！" << std::endl;
        return {0, 0, 0, 0, 0, {}, 0};
    }

    GDALRasterBand* poBand = poDataset->GetRasterBand(1);
    
    double adfGeoTransform[6];
    poDataset->GetGeoTransform(adfGeoTransform);
    
    double cell_size_x = adfGeoTransform[1]; 
    double cell_size_y = std::abs(adfGeoTransform[5]); 

    // 自适应检查水平单位是否为“度”
    bool is_geographic = (cell_size_x < 0.1);
    if (is_geographic) {
        std::cout << "[⚠️ 提示] 检测到地图水平单位为经纬度(度)。已自动启用重庆区域(北纬30°)米制投影系数修正！" << std::endl;
        cell_size_x = cell_size_x * 96000.0;  
        cell_size_y = cell_size_y * 111000.0; 
    }
    std::cout << "实际计算采用的像素物理尺寸: X = " << cell_size_x << "米, Y = " << cell_size_y << "米" << std::endl;

    int read_col = std::max(0, col_offset - 1);
    int read_row = std::max(0, row_offset - 1);
    int read_w = std::min(poDataset->GetRasterXSize() - read_col, width + 2);
    int read_h = std::min(poDataset->GetRasterYSize() - read_row, height + 2);

    std::vector<float> dem_data(read_w * read_h);
    CPLErr err = poBand->RasterIO(GF_Read, read_col, read_row, read_w, read_h, 
                                  dem_data.data(), read_w, read_h, GDT_Float32, 0, 0);

    if (err == CE_Failure) {
        std::cerr << "错误：读取数据缓冲区失败！" << std::endl;
        GDALClose(poDataset);
        return {0, 0, 0, 0, 0, {}, 0};
    }

    auto get_elev = [&](int c, int r) -> float {
        return dem_data[r * read_w + c];
    };

    double min_elev = 99999.0;
    double max_elev = -99999.0;
    double total_slope = 0.0;
    double max_slope = 0.0;
    long long valid_points = 0;

    // 初始化直方图区间 (5, 10, 15...90)
    std::map<int, long long> slope_bins;
    for (int i = 5; i <= 90; i += 5) {
        slope_bins[i] = 0;
    }

    // 开始遍历矩阵
    for (int r = 1; r < read_h - 1; ++r) {
        for (int c = 1; c < read_w - 1; ++c) {
            float z5 = get_elev(c, r);
            
            // 过滤无效高程
            if (z5 <= -100 || z5 > 9000) continue; 

            // 提取完整的周围8邻域点（修复报错的关键！）
            float z1 = get_elev(c - 1, r - 1);
            float z2 = get_elev(c, r - 1);
            float z3 = get_elev(c + 1, r - 1);
            float z4 = get_elev(c - 1, r);
            float z6 = get_elev(c + 1, r);
            float z7 = get_elev(c - 1, r + 1);
            float z8 = get_elev(c, r + 1);
            float z9 = get_elev(c + 1, r + 1);

            // 过滤邻域中带有无效点的数据块
            if (z1 <= -100 || z2 <= -100 || z3 <= -100 || z4 <= -100 || 
                z6 <= -100 || z7 <= -100 || z8 <= -100 || z9 <= -100) continue;

            if (z5 < min_elev) min_elev = z5;
            if (z5 > max_elev) max_elev = z5;

            // 计算梯度与坡度
            double dx = ((z3 + 2.0 * z6 + z9) - (z1 + 2.0 * z4 + z7)) / (8.0 * cell_size_x);
            double dy = ((z7 + 2.0 * z8 + z9) - (z1 + 2.0 * z2 + z3)) / (8.0 * cell_size_y);

            double rise_run = std::sqrt(dx * dx + dy * dy);
            double slope_deg = std::atan(rise_run) * 57.29578; 

            if (slope_deg > max_slope) max_slope = slope_deg;
            total_slope += slope_deg;
            valid_points++;

            // 直方图分类归档
            int bin_key = (static_cast<int>(slope_deg) / 5) * 5 + 5;
            if (bin_key > 90) bin_key = 90; 
            slope_bins[bin_key]++;
        }
    }

    GDALClose(poDataset);

    MountainMetrics result;
    result.min_elevation = (min_elev > 9000) ? 0 : min_elev;
    result.max_elevation = (max_elev < -100) ? 0 : max_elev;
    result.relative_height = result.max_elevation - result.min_elevation;
    result.max_slope = max_slope;
    result.avg_slope = (valid_points > 0) ? (total_slope / valid_points) : 0.0;
    result.slope_distribution = slope_bins;
    result.total_valid_points = valid_points;

    return result;
}

int main() {
    // 替换为您在 Codespace 工作区里上传的真实 700MB+ Tiff 文件名
    std::string dem_file = "chongqing_dem.tif"; 

    GDALAllRegister();
    GDALDataset* poDataset = (GDALDataset*)GDALOpen(dem_file.c_str(), GA_ReadOnly);
    if (poDataset == nullptr) {
        std::cerr << "无法打开文件，请检查 main 函数中的文件名！" << std::endl;
        return -1;
    }

    int img_width = poDataset->GetRasterXSize();
    int img_height = poDataset->GetRasterYSize();
    GDALClose(poDataset);

    std::cout << "--- 本地数据读取成功 ---" << std::endl;
    std::cout << "地图实际分辨率: " << img_width << " x " << img_height << " 像素" << std::endl;

    // 智能动态中心圈画
    int width = img_width * 0.10;
    int height = img_height * 0.10;
    int user_col = (img_width - width) / 2;  
    int user_row = (img_height - height) / 2; 

    std::cout << "智能圈画中心分析区域: 起点(" << user_col << ", " << user_row 
              << ")，窗口大小(" << width << " x " << height << ")" << std::endl;
    std::cout << "正在分析圈画区域山体特征...\n" << std::endl;
    
    MountainMetrics metrics = AnalyzeMountain(dem_file, user_col, user_row, width, height);

    std::cout << "=== 基础特征报告 ===" << std::endl;
    std::cout << "区域最高海拔: " << metrics.max_elevation << " 米 | 最低海拔: " << metrics.min_elevation << " 米" << std::endl;
    std::cout << "山体最大比高: " << metrics.relative_height << " 米" << std::endl;
    std::cout << "区域平均坡度: " << metrics.avg_slope << " 度" << std::endl;
    std::cout << "区域最大坡度: " << metrics.max_slope << " 度" << std::endl;

    std::cout << "\n=== 反向分析：山体坡度空间分布矩阵 ===" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(15) << "坡度区间" 
              << std::setw(15) << "像素点数量(面积)" 
              << std::setw(12) << "地形占比" 
              << "可视化分布图" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    for (const auto& pair : metrics.slope_distribution) {
        int upper_bound = pair.first;
        long long count = pair.second;
        
        double percentage = (metrics.total_valid_points > 0) ? 
                            (static_cast<double>(count) / metrics.total_valid_points * 100.0) : 0.0;

        std::string range_label = std::to_string(upper_bound - 5) + " - " + std::to_string(upper_bound) + " 度";
        std::cout << std::left << std::setw(15) << range_label 
                  << std::setw(15) << count 
                  << std::fixed << std::setprecision(2) << std::setw(10) << percentage << "%  ";

        int bar_length = static_cast<int>(percentage / 2.0);
        for (int i = 0; i < bar_length; ++i) std::cout << "*";
        std::cout << std::endl;
    }
    std::cout << "--------------------------------------------------------" << std::endl;

    return 0;
}