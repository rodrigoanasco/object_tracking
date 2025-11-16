#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>


int main() {
    
    // Model gets imported in ONNX form and its stored into a "net" (that stands for Neural Network)
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../src/yolov8n.onnx"); //import works fine!
    
    //video reader - working fine
    cv::VideoCapture video("../videos/walking_people.mp4");
    if (video.isOpened() == false){
        std::cout << "Cannot Open Video" << std::endl;
        return -1;
    }

    cv::Mat frame, gray;
    while (true){
        bool ret = video.read(frame);

        if(!ret || frame.empty()){
            //This can be triggered either when the video ends or if there is an error when reading
            std::cout << "Don't have any frame or can't read the video. Exiting the loop" << std::endl;
            break;
        }

        //convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        //display
        cv::imshow("frame", gray);

        if(cv::waitKey(25) == 'q'){
            break;
        }
    }
    std::cout << "Program Working Fine" << std::endl;
    
    return 0;
}