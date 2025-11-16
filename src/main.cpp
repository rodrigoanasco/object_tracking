#include <iostream>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/mat.hpp>
#include <opencv4/opencv2/core/operations.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/videoio.hpp>
#include <string>
#include <vector>


/*
YOLOv8 expects images of a fixed size during inference (by default 640x640)
So before sending the frame to the neural network, we do:
cv::dnn::blobFromImage(frame, ..., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), ...)
which resizes the original frame (e.g., 1920×1080) to 640×640.

Later, when we get box coordinates back (still in this 640×640 “network space”), we rescale them to the original frame size using x_factor and y_factor:
float x_factor = frame.cols / (float)INPUT_WIDTH;
float y_factor = frame.rows / (float)INPUT_HEIGHT;

These two constants describe the input resolution for the model, and are used both for preprocessing (resize) and for mapping detection coordinates back to the original image.
*/
const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;


const float CONFIDENCE_THRESHOLD = 0.70;
const float SCORE_THRESHOLD = 0.70;
const float NMS_THRESHOLD = 0.75;



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



        //interface
        cv::Mat output;
        net.forward(output);
        //YOLO normally gives the shape of [1, 84, 8400]
        //We'll reshape to [8400, 84] : 8400 predictions, 84 values each
        int rows = output.size[2]; // N = number of predictions
        int cols = output.size[1]; // C = 4 + 1 + num_classes
        cv::Mat output_reshaped(rows, cols, CV_32F, output.ptr<float>());

        int num_classes = cols - 5;

        //Post process: decode detection
        std::vector<int> class_ids;
        std::vector<float> confidences; 
        std::vector<cv::Rect> boxes;

        for(int i = 0; i < rows; i++){
            float box_confidence = output_reshaped.at<float>(i, 4);
            if (box_confidence < CONFIDENCE_THRESHOLD){
                continue;
            }

            cv::Mat scores = output_reshaped.row(i).colRange(5, 5 + num_classes);
            cv::Point class_id_point;
            double max_class_score;
            cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);

            float score = (float)max_class_score * box_confidence;
            if(score < SCORE_THRESHOLD){
                continue;
            }

            //Bounding box is (cx, cy, w, h) in the 640x640 space
            float cx = output_reshaped.at<float>(i, 0);
            float cy = output_reshaped.at<float>(i, 1);
            float w  = output_reshaped.at<float>(i, 2);
            float h  = output_reshaped.at<float>(i, 3);

            float x = (cx - 0.5f * w) * x_factor;
            float y = (cy - 0.5f * h) * y_factor;
            float width  = w * x_factor;
            float height = h * y_factor;

            class_ids.push_back(class_id_point.x);
            confidences.push_back(score);
            boxes.emplace_back((int)x, (int)y, (int)width, (int)height);
        }

        //NMS
        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_indices);

        //Draw results
        for (int idx : nms_indices) {
            cv::rectangle(frame, boxes[idx], cv::Scalar(0, 255, 0), 2);
            std::string label = std::to_string(class_ids[idx]) + " : " + cv::format("%.2f", confidences[idx]);
            cv::putText(frame, label, boxes[idx].tl() + cv::Point(0, -5), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        

        cv::imshow("DETECTIONS", frame);
        if(cv::waitKey(25) == 'q'){
            break;
        }
    }

    video.release();
    cv::destroyAllWindows();


    std::cout << "Program Working Fine" << std::endl;

    return 0;
}