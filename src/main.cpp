#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>


const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;
const float CONFIDENCE_THRESHOLD = 0.25;
const float SCORE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const int NUM_CLASSES = 80;


int main() {
    
    // Model gets imported in ONNX form and its stored into a "net" (that stands for Neural Network)
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../src/yolov8n.onnx"); //import works fine!
    
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);


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

        float x_factor = frame.cols / (float)INPUT_WIDTH;
        float y_factor = frame.rows / (float)INPUT_HEIGHT;
        
        //Pre-process
        cv::Mat blob;
        cv::dnn::blobFromImage(
            frame, blob,
            1.0/255.0,
            cv::Size(INPUT_WIDTH, INPUT_HEIGHT),
            cv::Scalar(), true, false
        );
        net.setInput(blob);

        //convert to grayscale
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        //display
        cv::imshow("frame", gray);

        if(cv::waitKey(25) == 'q'){
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();


    std::cout << "Program Working Fine" << std::endl;

    return 0;
}