#include <iostream>
#include <vector>
#include <cmath>

#include "utils.hpp"

int main() {
    // Считываем входное изображение
    cv::Mat realImg = cv::imread("image.jpg");
    cv::Mat inputImage = cv::imread("image.jpg", cv::IMREAD_GRAYSCALE);

    // Convert the input image to a 2D float vector
    std::vector<std::vector<float>> inputVec(inputImage.rows, std::vector<float>(inputImage.cols));
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            inputVec[i][j] = static_cast<float>(inputImage.at<uchar>(i, j));
        }
    }

    // Blur the input image
    std::vector<std::vector<float>> outputVec(inputImage.rows, std::vector<float>(inputImage.cols));
    GaussFilter::GaussianBlur(inputVec, outputVec, 5, 1.0);

    // cv::Mat outputImage(inputImage.rows, inputImage.cols, CV_8UC1);
    // for (int i = 0; i < inputImage.rows; i++) {
    //     for (int j = 0; j < inputImage.cols; j++) {
    //         outputImage.at<uchar>(i, j) = static_cast<uchar>(outputVec[i][j]);
    //     }
    // }

    std::vector<std::vector<float>> grad = sobelOperator(outputVec);
    std::vector<std::vector<uchar>> uGrad(grad.size(), std::vector<uchar>(grad[0].size(), 0));
    convertScaleAbs(grad, uGrad);

    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            grad[i][j] = static_cast<float>(uGrad[i][j]);
            // outputImage.at<uchar>(i, j) = static_cast<uchar>(outputVec[i][j]);
        }
    }

    std::vector<std::vector<int>> res(grad.size(), std::vector<int>(grad[0].size(), 0));
    Binarization::OtsuThreshold(grad, res);

    std::vector<std::vector<int>> labels(res.size(), std::vector<int>(res[0].size(), 0));
    Borders::CCA(res, labels);

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
    std::cout << "Количество меток:" << labelsCoords.size() << std::endl;

    for (const std::vector<std::pair<int, int>>& labs : labelsCoords)
    {
        if(labs.size() < 5)
        {
            labelsCoords.erase(std::remove(labelsCoords.begin(), labelsCoords.end(), labs), labelsCoords.end());
        }
    }
    std::cout << "Количество меток:" << labelsCoords.size() << std::endl;
    
    std::vector<std::tuple<int, int, int, int>> boundingBoxes;
    for (auto contour : labelsCoords) {
        int minX, minY, maxX, maxY;
        Borders::GetBoundingBox(contour, minX, minY, maxX, maxY);
        boundingBoxes.push_back(std::make_tuple(minX, minY, maxX, maxY));
    }

    cv::Mat gradImg(inputImage.rows, inputImage.cols, CV_8UC1);
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            gradImg.at<uchar>(i, j) = static_cast<uchar>(res[i][j]);
        }
    }

    for (const auto& rect: boundingBoxes)
    {   
        int x, y, endX, endY;
        x = std::get<1>(rect);
        y = std::get<0>(rect);
        endX = std::get<3>(rect);
        endY = std::get<2>(rect);

        cv::Rect rct(x, y, endX - x, endY - y);
        cv::Mat image = gradImg(rct);
        float result = std::sqrt(std::pow(endX - x, 2) + std::pow(endY - y, 2));
        float cons = countPixConcentration(image);
        if(result > 5 && endY - y > 3 && endX - x > 3)
            if (cons > 0.20)
                drawRectangle(realImg, std::get<1>(rect), std::get<0>(rect), std::get<3>(rect), std::get<2>(rect));
    }

    // Display the input and output images
    // cv::imshow("Input Image", inputImage);
    // cv::imshow("Output Image", outputImage);
    cv::imshow("RealImg", realImg);
    cv::imshow("Grad Image", gradImg);
    cv::waitKey(0);

    return 0;
}
