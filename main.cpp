#include <iostream>
#include <vector>
#include <cmath>

#include "utils.hpp"

int main() {
    // Считываем входное изображение
    cv::Mat realImg = cv::imread("image.jpg");
    cv::Mat inputImage = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);

    // Конвертируем входное изображение в двумерный массив
    std::vector<std::vector<float>> inputVec(inputImage.rows, std::vector<float>(inputImage.cols));
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            inputVec[i][j] = static_cast<float>(inputImage.at<uchar>(i, j));
        }
    }

    // Применяем размытие по Гауссу
    std::vector<std::vector<float>> outputVec(inputImage.rows, std::vector<float>(inputImage.cols));
    GaussFilter::GaussianBlur(inputVec, outputVec, 5, 1.0);

    // Вычисляем значения градиентов для каждого пикселя изображения
    std::vector<std::vector<float>> grad = sobelOperator(outputVec);

    // Конвертируем все значения в положительные
    std::vector<std::vector<uchar>> uGrad(grad.size(), std::vector<uchar>(grad[0].size(), 0));
    convertScaleAbs(grad, uGrad);

    // Выполняем бинаризацию изображения методом Оцу
    Binarization::OtsuThreshold(uGrad, uGrad);

    // Находим все объекты(области) на изображении, с помощью связного компонентного анализа
    std::vector<std::vector<int>> labels(uGrad.size(), std::vector<int>(uGrad[0].size(), 0));
    Borders::CCA(uGrad, labels);

    // Находим координаты точек, полученных объектов(областей)
    std::vector<std::vector<std::pair<int, int>>> labelsCoords;

    for (int i = 0; i < labels.size(); i++)
    {
        for (int j = 0; j < labels[0].size(); j++)
        {
            if (labels[i][j] > labelsCoords.size())
            {
                labelsCoords.push_back(std::vector<std::pair<int, int>>());
            }
            if(labels[i][j] != 0)
            {
                labelsCoords[labels[i][j] - 1].push_back(std::pair(i, j));
            }
        }
    }

    // Исключаем все объекты, число точек которых меньше 5
    for (const std::vector<std::pair<int, int>>& labs : labelsCoords)
    {
        if(labs.size() < 5)
        {
            labelsCoords.erase(std::remove(labelsCoords.begin(), labelsCoords.end(), labs), labelsCoords.end());
        }
    }
    
    // Находим координаты прямоугольников, которые будут ограничивать найденные области
    std::vector<std::tuple<int, int, int, int>> boundingBoxes;
    for (auto contour : labelsCoords) {
        int minX, minY, maxX, maxY;
        Borders::GetBoundingBox(contour, minX, minY, maxX, maxY);
        boundingBoxes.push_back(std::make_tuple(minX, minY, maxX, maxY));
    }

    // Переводим полученный массив пикселей интенсивности в изображение
    cv::Mat gradImg(inputImage.rows, inputImage.cols, CV_8UC1);
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            gradImg.at<uchar>(i, j) = uGrad[i][j];
        }
    }

    // Строим прямоугольники на исходном изображении
    for (const auto& rect: boundingBoxes)
    {   
        // Инициализируем координаты прямоугольника
        int x, y, endX, endY;
        x = std::get<1>(rect);
        y = std::get<0>(rect);
        endX = std::get<3>(rect);
        endY = std::get<2>(rect);

        // Вырезаем из бинаризованного изображения прямоугольники
        cv::Rect rct(x, y, endX - x, endY - y);
        cv::Mat image = gradImg(rct);

        // Вводим дополнительные проверки на длину диагоналей
        // и интенсивность белых пикселей на выбранном прямоугольнике
        // для отсечения ненужных значений
        float result = std::sqrt(std::pow(endX - x, 2) + std::pow(endY - y, 2));
        float cons = countPixConcentration(image);

        if(result > 5 && endY - y > 3 && endX - x > 3)
            if (cons > 0.20)
                drawRectangle(realImg, std::get<1>(rect), std::get<0>(rect), std::get<3>(rect), std::get<2>(rect));
    }

    // Выводим конечный результат готового изображения и бинаризованное изображение
    cv::imshow("RealImg", realImg);
    cv::imshow("Grad Image", gradImg);
    cv::waitKey(0);

    return 0;
}
