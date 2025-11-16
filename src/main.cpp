#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/videoio.hpp>


int main() {
    
    // Model gets imported in ONNX form and its stored into a "net" (that stands for Neural Network)
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../src/yolov8n.onnx"); //import works fine!
    
    //video reader - working fine
    cv::VideoCapture video("../videos/walking_people.mp4");
    if (video.isOpened() == false){
        std::cout << "Cannot Open Video" << std::endl;
    }

    
    std::cout << "Program Working Fine" << std::endl;

    return 0;
}