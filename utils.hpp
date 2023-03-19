#include <vector>
#include <cmath>
#include <limits.h>
#include <opencv2/opencv.hpp>

class Binarization
{
private:
    std::vector<int> ComputeHistogram(const std::vector<std::vector<uchar>>& image);
    std::vector<int> ComputeCumulativeSum(const std::vector<int>& input);
    float ComputeMeanIntensity(const std::vector<int>& histogram);
    float ComputeOtsuThreshold(const std::vector<std::vector<uchar>>& image);
    void BinaryThreshold(const std::vector<std::vector<uchar>>& inputImage, std::vector<std::vector<uchar>>& outputImage, float threshold);

    Binarization() {};

public:
    static void OtsuThreshold(const std::vector<std::vector<uchar>>& inputImage, std::vector<std::vector<uchar>>& outputImage);
};

class Borders
{
private:
    bool CheckBoundary(int x, int y, int rows, int cols);
    void BFS(int x, int y, int label, const std::vector<std::vector<uchar>>& binaryImg, std::vector<std::vector<int>>& labels);    
    
    Borders() {};

public:
    static void GetBoundingBox(const std::vector<std::pair<int, int>>& contour, int& minX, int& minY, int& maxX, int& maxY);
    static void CCA(const std::vector<std::vector<uchar>>& binaryImg, std::vector<std::vector<int>>& labels);
};

class GaussFilter
{
private:
    std::vector<std::vector<float>> CreateGaussianKernel(int kernelSize, float sigma);
    GaussFilter() {};
    
public:
    static void GaussianBlur(const std::vector<std::vector<float>>& inputImage, std::vector<std::vector<float>>& outputImage,
    int kernelSize, float sigma); 
};

std::vector<std::vector<float>> sobelOperator(std::vector<std::vector<float>>& image);

void convertScaleAbs(const std::vector<std::vector<float>>& image, std::vector<std::vector<uchar>>& res);

float countPixConcentration(cv::Mat& img);

void drawRectangle(cv::Mat& img, int x, int y, int endX, int endY);