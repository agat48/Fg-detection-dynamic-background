#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <list>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace cv;
using namespace std;


#define BUFFER_SIZE 30

int countDir(const char* path);
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
        double, const Scalar& color);
void calculateAverage(vector<Mat> in, Mat out, Mat tensor);
void calculateVariance(vector<Mat> in, Mat average, Mat tensor);

int main( int argc, const char** argv )
{
    const string absPath = "D:/Dokumenty/Inzynierka"; //"C:/Users/Agata/Desktop/Inzynierka";
    string path = absPath + "/git_repo/Fg-detection-dynamic-background";
    path = absPath + "/samples/dynamicBackground/dynamicBackground/";
    const string sampleNames[6] = {"boats","canoe","fall","fountain01","fountain02","overpass"};
    path = path + sampleNames[3] + "/input/";
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame, nextFrame, grayFrame, grayNextFrame, flow, cflow;
    vector<Mat> flowBuffer;
    namedWindow("MyWindow", CV_WINDOW_AUTOSIZE); //create a window with the name "MyWindow"
    sprintf(filename, "in%06d.jpg", 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    Mat average(frame.rows,frame.cols, CV_32FC2,0.0);
    Mat variance(average);
    Mat tensor(frame.rows,frame.cols, CV_32FC1,0.0);
    double min, max;
    for (int i = 1; i<sampleCount-1; i++) {
        sprintf(filename, "in%06d.jpg", i+1);
        nextFrame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);

        if (frame.empty()||nextFrame.empty()) //check whether the image is loaded or not
        {
            cout << "Error : Image cannot be loaded..!!" << endl;
            //system("pause"); //wait for a key press
            return -1;
        }
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        cvtColor(nextFrame, grayNextFrame, COLOR_BGR2GRAY);

        calcOpticalFlowFarneback(grayFrame, grayNextFrame, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
        cvtColor(grayFrame, cflow, COLOR_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 16, 1.5, Scalar(0, 255, 0));
//        split(flow, channels); //split image into separate channels
//        imshow("MyWindow", cflow); //display the image which is stored in the 'img' in the "MyWindow" window

//        waitKey(0); //wait infinite time for a keypress
        swap(frame, nextFrame);
        if(flowBuffer.size() >= BUFFER_SIZE) {
            flowBuffer.erase(flowBuffer.begin());
        }
        flowBuffer.push_back(flow);
        calculateAverage(flowBuffer, average, tensor);
        calculateVariance(flowBuffer, average, tensor);
        imshow("MyWindow", tensor);
        minMaxLoc(tensor, &min, &max); //minimalna i maksymalna wartość
        cout << min << " " << max << endl;
        waitKey(1);
    }
    destroyWindow("MyWindow"); //destroy the window with the name, "MyWindow"

    return 0;
}

int countDir(const char *path) {
    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir (path);

    if (dp != NULL)
    {
        while (ep = readdir (dp))
            i++;

        (void) closedir (dp);
    }
    else
        perror ("Couldn't open the directory");

    return i;
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
        double, const Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step) {
        for (int x = 0; x < cflowmap.cols; x += step) {
            const Point2f &fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x, y), Point(cvRound(x + fxy.x), cvRound(y + fxy.y)),
                    color);
            //circle(cflowmap, Point(x,y), 2, color, -1);
        }
    }
}

void calculateAverage(vector<Mat> in, Mat out, Mat tensor) {
    Point2f avg;
    for(int y = 0; y < in[0].rows; y++) {
        for (int x = 0; x < in[0].cols; x++) {
            avg.x = 0;
            avg.y = 0;
            for (int i = 0; i < in.size(); i++) {
                const Point2f &fxy = in[i].at<Point2f>(y, x);
                avg+=fxy;
            }
            avg.x/=in.size();
            avg.y/=in.size();
            out.at<Point2f>(y,x)= avg;

//            tensor.at<float>(y,x) = sqrt(avg.x + avg.y);
        }
    }
}

void calculateVariance(vector<Mat> in, Mat average, Mat tensor) {
    Point2f var;
    for(int y = 0; y < in[0].rows; y++) {
        for (int x = 0; x < in[0].cols; x++) {
            var.x = 0;
            var.y = 0;
            const Point2f &avg = average.at<Point2f>(y, x);
            for (int i = 0; i < in.size(); i++) {
                const Point2f &fxy = in[i].at<Point2f>(y, x);
                var.x+=pow(fxy.x-avg.x,2.0);
                var.y+=pow(fxy.y-avg.y,2.0);
            }
            var.x/=in.size();
            var.y/=in.size();

            tensor.at<float>(y,x) = sqrt(var.x + var.y)*10000000;
        }
    }
}