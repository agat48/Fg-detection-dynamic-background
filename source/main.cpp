#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>

using namespace cv;
using namespace std;


#define BUFFER_SIZE 20

int countDir(const char* path);
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
        double, const Scalar& color);
void calculateAverage(vector<Mat> in[3], vector<Mat> average);
void calculateVariance(vector<Mat> in[3], vector<Mat> average, Mat tensor);

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ );

int FTSG(string path);

Mat window;

int main( int argc, const char** argv )
{
    const string absPath = "D:/Dokumenty/Inzynierka"; //"C:/Users/Agata/Desktop/Inzynierka";
    string path = absPath + "/git_repo/Fg-detection-dynamic-background";
    path = absPath + "/samples/dynamicBackground/dynamicBackground/";
    const string sampleNames[6] = {"boats","canoe","fall","fountain01","fountain02","overpass"};
    path = path + sampleNames[0] + "/input/";
    FTSG(path);
    return 0;
}

int countDir(const char *path) {
    DIR *dp;
    int i = 0;
    dp = opendir (path);

    if (dp != NULL)
    {
        while (readdir (dp))
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

void calculateAverage(vector<Mat> in[3], vector<Mat> average) {
    Point2f avg;
    int sampleCount = in[0].size();
    int rows = in[0][0].rows;
    int cols = in[0][0].cols;
    for(int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            avg.x = 0;
            avg.y = 0;
            for (int k = 0; k < 3; k++) {
                for (int i = 0; i < sampleCount; i++) {
                    const Point2f &fxy = in[k][i].at<Point2f>(y, x);
                    avg += fxy;
                }
                avg.x/=sampleCount;
                avg.y/=sampleCount;
                average[k].at<Point2f>(y,x)= avg;
            }
        }
    }
}

void calculateVariance(vector<Mat> in[3], vector<Mat> average, Mat tensor) {
    Point2f var;
    int sampleCount = in[0].size();
    int rows = in[0][0].rows;
    int cols = in[0][0].cols;
    for(int y = 0; y < rows; y++) {
        for (int x = 0; x < cols; x++) {
            var.x = 0;
            var.y = 0;
            for (int k = 0; k < 3; k++) {
                const Point2f &avg = average[k].at<Point2f>(y, x);
                for (int i = 0; i < sampleCount; i++) {
                    const Point2f &fxy = in[k][i].at<Point2f>(y, x);
                    var.x += pow(fxy.x - avg.x, 2.0);
                    var.y += pow(fxy.y - avg.y, 2.0);
                }
            }
            var.x /= sampleCount;
            var.y /= sampleCount;
            tensor.at<float>(y,x) = sqrt(var.x + var.y);
        }
    }
}

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ )
{
    if( event == CV_EVENT_LBUTTONDOWN )
    {
        cout << window.at<Point2f>(y, x) << endl;
    }
}

int FTSG(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame, nextFrame, flow;
    vector<Mat> frameChannels, nextFrameChannels, flowChannels, tensorChannels;
    vector<Mat> flowBuffer[3]; //flow history for each channel
    BackgroundSubtractorMOG2 mog(BUFFER_SIZE, 16, false);
    namedWindow("oryg", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    namedWindow("Flux Tensor", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
//    setMouseCallback( "Flux Tensor", onMouse, 0 );
    namedWindow("Split Gaussian", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    namedWindow("output", CV_WINDOW_AUTOSIZE);
    sprintf(filename, "in%06d.jpg", 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    vector<Mat> average;
    Mat tensor(frame.rows,frame.cols, CV_32FC1,0.0);//, tensorMerge;
    Mat tensorBin, bgMaskBin, fgMaskBin;
    Mat moving;
    Mat bgMask, fgMask, amb, bg, bgDiff, output;
    double min, max;

    for (int i = 0; i < 3; i++) { //space in vectors
        flowChannels.emplace_back();
        average.emplace_back(frame.rows,frame.cols, CV_32FC2,0.0);
    }
    for (int i = 1; i<sampleCount-1; i++) { //loop over frames
        sprintf(filename, "in%06d.jpg", i+1);
        nextFrame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);

        if (frame.empty()||nextFrame.empty()) //check whether the image is loaded or not
        {
            cout << "Error : Image cannot be loaded..!!" << endl;
            //system("pause"); //wait for a key press
            return -1;
        }
        split(frame, frameChannels);
        split(nextFrame, nextFrameChannels);
        for(int i = 0; i < frameChannels.size(); i++) {
            calcOpticalFlowFarneback(frameChannels[i], nextFrameChannels[i], flowChannels[i], 0.5, 3, 15, 3, 5, 1.2, 0);
        }
//        drawOptFlowMap(flow[0], frame, 16, 1.5, Scalar(0, 255, 0));
//        split(flow, channels); //split image into separate channels
//        imshow("Flux Tensor", cflow); //display the image which is stored in the 'img' in the "Flux Tensor" window

//        waitKey(0); //wait infinite time for a keypress
        if(flowBuffer[0].size() >= BUFFER_SIZE) {
            for (int i = 0; i < 3; i++) {
                flowBuffer[i].erase(flowBuffer[i].begin());
            }
        }
        for (int i = 0; i < 3; i++) {
            flowBuffer[i].push_back(flowChannels[i]);
        }
        calculateAverage(flowBuffer, average);
        calculateVariance(flowBuffer, average, tensor);
//        for(int i = 0; i < 3; i++) {
//            calcCovarMatrix(flowBuffer[i], tensorChannels[i], average[i], 0);
//        }
//        minMaxLoc(tensor, &min, &max); //minimalna i maksymalna wartość
//        cout << min << " " << max << endl;
        mog(frame,bgMask,0.02);
//        merge(tensorChannels, tensorMerge);
//        cvtColor(tensorMerge, tensorBin, COLOR_BGR2GRAY);
        threshold(tensor, tensorBin, 0.09, 1, THRESH_BINARY);
        threshold(bgMask, bgMaskBin, 150, 1, THRESH_BINARY);
//        tensor.copyTo(window);
        bgMaskBin.convertTo(bgMaskBin, tensorBin.type());
//        moving = tensorBin & bgMaskMerge;
        bitwise_and(tensorBin, bgMaskBin, moving); //moving object detection
        bitwise_xor(tensorBin, bgMaskBin, amb); //ambiguous regions detection
        amb.convertTo(amb, tensorBin.type());
        bitwise_and(amb, bgMaskBin, amb);
        mog.getBackgroundImage(bg);
        bgDiff = abs(frame - bg);
        cvtColor(bgDiff, fgMask, COLOR_BGR2GRAY);
        threshold(fgMask, fgMaskBin, 15, 1, THRESH_BINARY);
        amb.convertTo(amb, fgMaskBin.type());
        bitwise_and(fgMaskBin, amb, fgMaskBin); //static foreground mask
        fgMaskBin.convertTo(fgMaskBin, moving.type());
        bitwise_or(moving, fgMaskBin, output);
        imshow("oryg", frame);
        imshow("Flux Tensor", tensorBin);
        imshow("Split Gaussian", bgMaskBin);
        imshow("output", output);
        waitKey(1);
        swap(frame, nextFrame);
    }
    destroyWindow("oryg"); //destroy the window with the name, "Flux Tensor"
    destroyWindow("Flux Tensor"); //destroy the window with the name, "Flux Tensor"
    destroyWindow("Split Gaussian"); //destroy the window with the name, "Split Gaussian"
    destroyWindow("output");

}