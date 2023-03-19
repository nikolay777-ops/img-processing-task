#include "utils.hpp"

std::vector<int> Binarization::ComputeHistogram(const std::vector<std::vector<uchar>>& image)
{
    std::vector<int> histogram(256, 0);

    for (const auto& row: image)
    {
        for (const auto& pixel: row)
        {
            int intensity = static_cast<int>(pixel);
            histogram[intensity]++;
        }
    }

    return histogram;
}

std::vector<int> Binarization::ComputeCumulativeSum(const std::vector<int>& input)
{
    std::vector<int> output(input.size());
    int sum = 0;

    for (int i = 0; i < input.size(); i++)
    {
        sum += input[i];
        output[i] = sum;
    }
    
    return output;
}

float Binarization::ComputeMeanIntensity(const std::vector<int>& histogram)
{
    float sum = 0.0f;
    float count = 0.0f;

    for (int i = 0; i < histogram.size(); i++)
    {
        sum += i * histogram[i];
        count += histogram[i];
    }

    return sum / count;
}

float Binarization::ComputeOtsuThreshold(const std::vector<std::vector<uchar>>& image)
{
    std::vector<int> histogram = Binarization::ComputeHistogram(image);
    std::vector<int> cumulativeSum = Binarization::ComputeCumulativeSum(histogram);

    int size = image.size() * image[0].size();
    float meanIntensity = Binarization::ComputeMeanIntensity(histogram);

    float maxVariance = 0.0f;
    float threshold = 0.0f;

    for (int i = 0; i < histogram.size(); i++)
    {
        float weightBackground = static_cast<float>(cumulativeSum[i]) / size;
        float weightForeground = 1.0 - weightBackground;

        float meanBackground = static_cast<float>(cumulativeSum[i] * i) / cumulativeSum[i];
        float meanForeground = (meanIntensity - cumulativeSum[i] * meanBackground / size) / weightForeground;
        
        float variance = weightBackground * weightForeground * std::pow((meanBackground - meanForeground), 2.0);

        if(variance > maxVariance)
        {
            maxVariance = variance;
            threshold = i;
        }
    }
    
    return threshold;
}

void Binarization::BinaryThreshold(const std::vector<std::vector<uchar>>& inputImage, std::vector<std::vector<uchar>>& outputImage, float threshold)
{
    for (int i = 0; i < inputImage.size(); i++)
    {
        for (int j = 0; j < inputImage[0].size(); j++)
        {
            outputImage[i][j] = (inputImage[i][j] >= threshold ? 255: 0);
        }
    }
}

void Binarization::OtsuThreshold(const std::vector<std::vector<uchar>>& inputImage, std::vector<std::vector<uchar>>& outputImage)
{
    Binarization bin;

    float threshold = bin.ComputeOtsuThreshold(inputImage);
    threshold /= 2.4;
    bin.BinaryThreshold(inputImage, outputImage, threshold);
}


// Функция для проверки пикселя на границы изображения
bool Borders::CheckBoundary(int x, int y, int rows, int cols) 
{
    return (x >= 0 && y >= 0 && x < rows && y < cols);
}

// Функция для выполнения поиска в ширину (BFS)
void Borders::BFS(int x, int y, int label, const std::vector<std::vector<uchar>>& binaryImg, std::vector<std::vector<int>>& labels) 
{
    int rows = binaryImg.size();
    int cols = binaryImg[0].size();

    std::vector<std::pair<int, int>> queue; // Очередь для BFS
    queue.push_back(std::make_pair(x, y));

    while (!queue.empty()) 
    {    
        int current_x = queue.front().first;
        int current_y = queue.front().second;
        queue.erase(queue.begin());

        // Проверяем соседние пиксели
        for (int i = -1; i <= 1; i++) 
        {
            for (int j = -1; j <= 1; j++) 
            {
                if (Borders::CheckBoundary(current_x + i, current_y + j, rows, cols) &&
                    binaryImg[current_x + i][current_y + j] == 255 &&
                    labels[current_x + i][current_y + j] == 0)
                    {
                        labels[current_x + i][current_y + j] = label;
                        queue.push_back(std::make_pair(current_x + i, current_y + j));
                    }
            }
        }
    }
}


// Функция для выполнения связного компонентного анализа (CCA)
void Borders::CCA(const std::vector<std::vector<uchar>>& binaryImg, std::vector<std::vector<int>>& labels) 
{
    int rows = binaryImg.size();
    int cols = binaryImg[0].size();
    int currentLabel = 0;
    
    Borders bord;

    // Проходим по каждому пикселю бинаризованного изображения
    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < cols; j++) 
        {
            if (binaryImg[i][j] == 255 && labels[i][j] == 0) {
                currentLabel++;
                labels[i][j] = currentLabel;
                bord.BFS(i, j, currentLabel, binaryImg, labels);
            }
        }
    }
}

void Borders::GetBoundingBox(const std::vector<std::pair<int, int>>& contour, int& minX, int& minY, int& maxX, int& maxY) {
    // Инициализация переменных координат
    minX = INT_MAX;
    minY = INT_MAX;
    maxX = INT_MIN;
    maxY = INT_MIN;

    // Поиск минимальных и максимальных координат
    for (auto point : contour) {
        if (point.first < minX) {
            minX = point.first;
        }
        if (point.second < minY) {
            minY = point.second;
        }
        if (point.first > maxX) {
            maxX = point.first;
        }
        if (point.second > maxY) {
            maxY = point.second;
        }
    }
}


// Функция для создания ядря свёртки Гаусса
std::vector<std::vector<float>> GaussFilter::CreateGaussianKernel(int kernelSize, float sigma) {
    std::vector<std::vector<float>> kernel(kernelSize, std::vector<float>(kernelSize, 0.0));
    float s = 2 * sigma * sigma;
    float sum = 0.0;
    int radius = kernelSize / 2;

    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            float r = std::sqrt(x * x + y * y);
            int i = x + radius;
            int j = y + radius;
            kernel[i][j] = (std::exp(-(r * r) / s)) / (M_PI * s);
            sum += kernel[i][j];
        }
    }

    // После создания ядра, выполним нормализацию, для получения значений в промежутке [0, 1]
    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

// Функция для выполнения размытия по Гауссу
void GaussFilter::GaussianBlur(const std::vector<std::vector<float>>& inputImage, std::vector<std::vector<float>>& outputImage,
    int kernelSize, float sigma) 
{
    GaussFilter gF;
    // Создадим ядро свёртки
    std::vector<std::vector<float>> kernel = gF.CreateGaussianKernel(kernelSize, sigma);

    int height = inputImage.size();
    int width = inputImage[0].size();
    int radius = kernelSize / 2;

    // Цикл для прохождения по всем пикселям изображения
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0;
            float weight = 0.0;

            // Цикл для прохождения по матрице свёртки
            for (int k = -radius; k <= radius; k++) {
                for (int l = -radius; l <= radius; l++) {
                    int x = i + k;
                    int y = j + l;

                    // Применяем ядро, если Х и Y находятся в диапазоне изображения
                    if (x >= 0 && x < height && y >= 0 && y < width) {
                        sum += kernel[k + radius][l + radius] * inputImage[x][y];
                        weight += kernel[k + radius][l + radius];
                    }
                }
            }

            // Сохраняем полученные значения
            outputImage[i][j] = sum / weight;
        }
    }
}

// Функция для расчёта градиента изображения с помощью оператора Собеля
std::vector<std::vector<float>> sobelOperator(std::vector<std::vector<float>>& image)
{   
    // Создаём ядро оператора Собеля для оси Х и оси Y
    std::vector<std::vector<int>> kernelX = {{1, 0, -1}, {2, 0, -2}, {1, 0, -1}};
    std::vector<std::vector<int>> kernelY = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // Создаём переменные радиуса ядра, значений ширины и высоты изображения 
    int radius = kernelX.size() / 2;
    int height = image.size();
    int width = image[0].size();
    
    std::vector<std::vector<float>> res(height, std::vector<float>(width, 0));

    // Цикл для прохождения по каждому пикселю изображения
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float gradX = 0.0f;
            float gradY = 0.0f;
            
            // Цикл для применения ядра оператора Собеля
            for (int k = -radius; k <= radius; k++)
            {
                for (int l = -radius; l <= radius; l++)
                {
                    int x = i + k;
                    int y = j + l;

                    if (x >= 0 && x < height && y >= 0 && y < width)
                    {
                        gradX += kernelX[k + radius][l + radius] * image[x][y];
                        gradY += kernelY[k + radius][l + radius] * image[x][y];
                    }
                }
            }
            // Расчёт значения градиента для каждого пикселя
            res[i][j] = std::sqrt(gradX * gradX + gradY * gradY);
        }
    }
    
    return res;
}

void convertScaleAbs(const std::vector<std::vector<float>>& image, std::vector<std::vector<unsigned char>>& res)
{
    float alpha = 255.0 / (image[0][0] + 1e-6);
    int height = image.size();
    int width = image[0].size();

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            res[i][j] = static_cast<unsigned char>(image[i][j] * alpha);
        }
    }
}

float countPixConcentration(cv::Mat& img)
{
    unsigned char* img_data = img.data;
    
    cv::Size sizes = img.size();

    unsigned int count = 0;

    for (int k = 0; k < sizes.height; k++) {
        for (int j = 0; j < sizes.width; j++) {
            if (img_data[k * img.step + j] == 255)
                count++;
        }
    }

    return float(count) / (sizes.height * sizes.width);
}


void drawRectangle(cv::Mat& img, int x, int y, int endX, int endY)
{
    cv::Point p1(x, y);
    cv::Point p2(endX, endY);

    rectangle(img, p1, p2, 
    cv::Scalar(0, 0, 255), 1, cv::LINE_8);
}