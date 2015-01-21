#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <dirent.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/video/background_segm.hpp>
#include <windef.h>
#include "bgfg_vibe.hpp"
#include "PBAS.h"

using namespace cv;
using namespace std;


#define BUFFER_SIZE 30
#define OMITTED_FRAMES 0

int countDir(const char* path);
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,
        double, const Scalar& color);
void calculateAverage(vector<Mat> in[3], vector<Mat> average);
void calculateVariance(vector<Mat> in[3], vector<Mat> average, Mat tensor);

static void onMouse( int event, int x, int y, int /*flags*/, void* /*param*/ );

void calculateAddress(Vec3b ptr, ushort map[192][3], ushort address[36]);

int FTSG(string path);
int CwisarDH(string path);
int GMM(string path);
int ViBE(string path);
int PBASfun(string path);

Mat window;

int main( int argc, const char** argv )
{
    const string absPath = "D:/Dokumenty/Inzynierka"; //"C:/Users/Agata/Desktop/Inzynierka";
    string path = absPath + "/git_repo/Fg-detection-dynamic-background";
    const string datasets[2] = {"dynamicBackground", "baseline"};
    const string sampleNames[2][6] = {{"boats","canoe","fall","fountain01","fountain02","overpass"},{"highway", "office", "pedestrians", "PETS2006"}};
    int mode = 0;
    string basicPath = absPath + "/samples/" + datasets[mode] + "/" + datasets[mode] + "/";
    for (int i = 0; i < 6; i++) {
        path = basicPath + sampleNames[mode][i] + "/input/";
//        FTSG(path);
        CwisarDH(path);
    }
    mode = 1;
    basicPath = absPath + "/samples/" + datasets[mode] + "/" + datasets[mode] + "/";
    for (int i = 0; i < 4; i++) {
        path = basicPath + sampleNames[mode][i] + "/input/";
        FTSG(path);
        CwisarDH(path);
    }
//    GMM(path);
//    ViBE(path);
//    PBASfun(path);

    return 0;
};

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

void calculateAddress(Vec3b ptr, ushort map[192][3], ushort address[36]) {
    ushort label;
    for (int i = 0; i < 36; i++) {
        address[i] = 0;
    }
    Vec3b val(ptr);
    for (int k = 0; k < 3; k++) {
//        val[k] = (val[k] <192 ? val[k] : 192);
        val[k] = val[k] * 192 / 255;
    }
    for (int l = 0; l < 192; l++) { //pixel retina construct
        for (int k = 0; k < 3; k++) { //for ones on the retina...
            label = map[l][k]; //calculating addresses for each label
            if (l < val[k]) {
                address[label] <<= 1;
                address[label] |= (ushort)1;
            }
            else {
                address[label] <<= 1;
            }
        }
    }
}

int GMM(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame, bgMask, bgMaskOpen;
    BackgroundSubtractorMOG2 mog(3, 16, false);
    sprintf(filename, "in%06d.jpg", OMITTED_FRAMES + 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    Mat strel = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));
    namedWindow("input", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    namedWindow("Split Gaussian", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    for (int i = OMITTED_FRAMES + 1; i < sampleCount; i++) {
        mog(frame,bgMask,0.001);
        sprintf(filename, "in%06d.jpg", i);
        frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
        morphologyEx(bgMask, bgMaskOpen, MORPH_OPEN, strel);
        imshow("input", frame);
        imshow("Split Gaussian", bgMaskOpen);
        waitKey(1);
//        cout << endl << i << endl;
    }
    destroyWindow("Split Gaussian"); //destroy the window with the name, "Split Gaussian"
    destroyWindow("input"); //destroy the window with the name, "Split Gaussian"
}

int ViBE(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame;
    sprintf(filename, "in%06d.jpg", OMITTED_FRAMES + 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    bgfg_vibe bgfg;
    bgfg.init_model(frame);
    namedWindow("input", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    namedWindow("output", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    for (int i = OMITTED_FRAMES + 1; i < sampleCount; i++) {
        {
            sprintf(filename, "in%06d.jpg", i);
            frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
            Mat fg = *bgfg.fg(frame);
            imshow("input", frame);
            imshow("output", fg);
            waitKey(1);
        }
        cout << i << endl;
    }
    destroyWindow("output"); //destroy the window with the name, "Split Gaussian"
    destroyWindow("input"); //destroy the window with the name, "Split Gaussian"
    return 0;

}

int PBASfun(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame;
    sprintf(filename, "in%06d.jpg", OMITTED_FRAMES + 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    PBAS pbas;
    Mat bluredImage;
    Mat pbasResult;


    namedWindow("input", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    namedWindow("output", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
    for (int i = OMITTED_FRAMES + 1; i < sampleCount; i++) {
        {
            GaussianBlur(frame, bluredImage, cv::Size(5,5), 1.5);
            pbas.process(&bluredImage, &pbasResult);
            medianBlur(pbasResult, pbasResult, 5);
            imshow("input", frame);
            imshow("output", pbasResult);
            waitKey(1);
            sprintf(filename, "in%06d.jpg", i);
            frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
        }
        cout << i << endl;
    }
    destroyWindow("output"); //destroy the window with the name, "Split Gaussian"
    destroyWindow("input"); //destroy the window with the name, "Split Gaussian"
    return 0;
}


int FTSG(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame, nextFrame, flow;
    vector<Mat> frameChannels, nextFrameChannels, flowChannels;
    vector<Mat> flowBuffer[3]; //flow history for each channel
    BackgroundSubtractorMOG2 mog(3, 16, false);
    sprintf(filename, "in%06d.jpg", OMITTED_FRAMES + 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    vector<Mat> average;
    Mat tensor(frame.rows,frame.cols, CV_32FC1,0.0);//, tensorMerge;
    Mat tensorBin, bgMaskBin, movingOpen, fgMaskBin;
    Mat moving;
    Mat bgMask, fgMask, amb, bg, bgDiff, output;
    Mat strel = getStructuringElement(MORPH_ELLIPSE, Size(5,5), Point(2,2));
    double min, max;

    for (int i = 0; i < 3; i++) { //space in vectors
        flowChannels.emplace_back();
        average.emplace_back(frame.rows,frame.cols, CV_32FC2,0.0);
    }
    for (int i = OMITTED_FRAMES + 1; i < BUFFER_SIZE + OMITTED_FRAMES; i++) { //loop over frames
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
        for(int k = 0; k < 3; k++) {
            calcOpticalFlowFarneback(frameChannels[k], nextFrameChannels[k], flowChannels[k], 0.5, 3, 15, 3, 5, 1.2, 0);
        }
//        drawOptFlowMap(flow[0], frame, 16, 1.5, Scalar(0, 255, 0));
//        split(flow, channels); //split image into separate channels
//        imshow("Flux Tensor", cflow); //display the image which is stored in the 'img' in the "Flux Tensor" window

//        waitKey(0); //wait infinite time for a keypress
        for (int k = 0; k < 3; k++) {
            flowBuffer[k].push_back(flowChannels[k]);
        }
//        calculateAverage(flowBuffer, average);
//        calculateVariance(flowBuffer, average, tensor);
////        minMaxLoc(tensor, &min, &max); //minimalna i maksymalna wartość
////        cout << min << " " << max << endl;
        mog(frame,bgMask,0.004);
        swap(frame, nextFrame);
//        cout << endl << i << endl;
    }

//    namedWindow("oryg", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
//    namedWindow("Flux Tensor", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
//    namedWindow("Split Gaussian", CV_WINDOW_AUTOSIZE); //create a window with the name "Flux Tensor"
//    namedWindow("output", CV_WINDOW_AUTOSIZE);
    long int start = getTickCount();
    for (int i = BUFFER_SIZE + OMITTED_FRAMES; i < 100; i++) { //loop over frames
        sprintf(filename, "in%06d.jpg", i+1);
        nextFrame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);

        if (frame.empty()||nextFrame.empty()) //check whether the image is loaded or not
        {
            cout << "Error : Image cannot be loaded..!!" << endl;
            return -1;
        }
        split(frame, frameChannels);
        split(nextFrame, nextFrameChannels);
        for(int k = 0; k < 3; k++) {
            calcOpticalFlowFarneback(frameChannels[k], nextFrameChannels[k], flowChannels[k], 0.5, 3, 15, 3, 5, 1.2, 0);
        }
            for (int k = 0; k < 3; k++) {
                flowBuffer[k].erase(flowBuffer[k].begin());
                flowBuffer[k].push_back(flowChannels[k]);
            }
        calculateAverage(flowBuffer, average);
        calculateVariance(flowBuffer, average, tensor);
        mog(frame,bgMask,0.004);
        threshold(tensor, tensorBin, 0.036, 1, THRESH_BINARY);
        threshold(bgMask, bgMaskBin, 150, 1, THRESH_BINARY);
        bgMaskBin.convertTo(bgMaskBin, tensorBin.type());
        bitwise_and(tensorBin, bgMaskBin, moving); //moving object detection
        bitwise_xor(tensorBin, bgMaskBin, amb); //ambiguous regions detection
        amb.convertTo(amb, tensorBin.type());
        bitwise_and(amb, bgMaskBin, amb);
        mog.getBackgroundImage(bg);
        bgDiff = abs(frame - bg);
        cvtColor(bgDiff, fgMask, COLOR_BGR2GRAY);
        threshold(fgMask, fgMaskBin, 2, 1, THRESH_BINARY);
        amb.convertTo(amb, fgMaskBin.type());
        bitwise_and(fgMaskBin, amb, fgMaskBin); //static foreground mask
        fgMaskBin.convertTo(fgMaskBin, moving.type());
        bitwise_or(moving, fgMaskBin, output);
        morphologyEx(moving, movingOpen, MORPH_OPEN, strel);
//        imshow("oryg", frame);
//        imshow("Flux Tensor", tensorBin);
//        imshow("Split Gaussian", bgMaskBinOpen);
//        imshow("output", output);
//        waitKey(1);
//        sprintf(filename, "out%06d.jpg", i);
//        movingOpen = 255 * movingOpen;
//        imwrite(path + "outputFTSG/" + filename, movingOpen);
        swap(frame, nextFrame);
    }
//    waitKey(0);
//    destroyWindow("oryg"); //destroy the window with the name, "Flux Tensor"
//    destroyWindow("Flux Tensor"); //destroy the window with the name, "Flux Tensor"
//    destroyWindow("Split Gaussian"); //destroy the window with the name, "Split Gaussian"
//    destroyWindow("output");

    long int end = getTickCount();
    cout << path << " FTSG time: " << end-start << endl;
    return 0;

}
int CwisarDH(string path) {
    int sampleCount = countDir(path.c_str()) - 2;
    char filename[12];
    Mat frame;
    sprintf(filename, "in%06d.jpg", 1);
    frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
    int width = frame.cols, height = frame.rows, size = pow(2,12);
    if(width * height > 200000) {
        cout << "Not enough memory" << endl;
        return -1;
    }
    Mat out(height, width, CV_8UC1), outOpen;
    Mat strel = getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));
    vector<Vec3b>** histBuff;
    ushort ***discr;
    discr = new ushort**[height];
    histBuff = new vector<Vec3b>*[height];
    int histBuffSize = 100;
    for (int i = 0; i < height; i++) {
        discr[i] = new ushort*[width];
        histBuff[i] = new vector<Vec3b>[width];
        for (int j = 0; j < width; j++) {
            discr[i][j] = new ushort[size];
            for (int k = 0; k < size; k++) {
                discr[i][j][k] = 0;
            }
        }
    }
    vector<ushort> values;
    for (int i = 0; i < 36; i++) {
        for (int j = 0; j < 16; j++) {
            values.push_back(i);
        }
    }
    ushort map[192][3];
    int range = values.size();
    for (int i = 0; i < 192; i++) {
        for (int j = 0; j < 3; j++) {
            int index = rand() % range;
            map[i][j] = values[index];
            values.erase(values.begin() + index);
            range--;
        }
    }

    for (int i = OMITTED_FRAMES; i < BUFFER_SIZE + OMITTED_FRAMES; i++) { // initialization
        ushort address[36];
        ushort ind, pos;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Vec3b ptr = frame.at<Vec3b>(y,x);
                calculateAddress(ptr, map, address);
                for (int k = 0; k < 36; k++) { //discriminator per-pixel learning
                    ind = address[k] >> 4;
                    pos = address[k] % 16;
                    discr[y][x][ind] |= (ushort)1 << pos;
                }
            }
        }
        sprintf(filename, "in%06d.jpg", i+1);
        frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
//        cout << endl << i << endl;
    }
//    namedWindow("input", CV_WINDOW_AUTOSIZE);
//    namedWindow("output", CV_WINDOW_AUTOSIZE);
    cout << "start" << endl;
    int64 st = getTickCount();
    long int start = getTickCount();
    for (int i = BUFFER_SIZE + OMITTED_FRAMES; i < 100; i++) {
        out = Mat::zeros(height, width, CV_8UC1);
        ushort address[36];
        ushort ind, pos;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                Vec3b ptr = frame.at<Vec3b>(y,x);
                calculateAddress(ptr, map, address);
                ushort similCount = 0;
                for (int k = 0; k < 36; k++) { //discriminator per-pixel learning
                    ind = address[k] >> 4;
                    pos = address[k] % 16;
                    if(discr[y][x][ind] & ((ushort)1 << pos) == (ushort)1 << pos) {
                        similCount++;
                    }
                }
                if (similCount < 29) {
                    out.at<uchar>(y,x) = 255;
                    histBuff[y][x].push_back(ptr);
                    if (histBuff[y][x].size() > histBuffSize) { //history buffer learning
                        for (int i = 0; i < histBuffSize; i++) {
                            Vec3b ptr = histBuff[y][x][i];
                            calculateAddress(ptr, map, address);
                            ushort ind, pos;
                            for (int k = 0; k < 36; k++) { //discriminator per-pixel learning
                                ind = address[k] >> 4;
                                pos = address[k] % 16;
                                discr[y][x][ind] |= (ushort)1 << pos;
                            }
                        }
                        histBuff[y][x].empty();
                    }
                }
                else {
                    for (int k = 0; k < 36; k++) { //discriminator per-pixel learning
                        ind = address[k] >> 4;
                        pos = address[k] % 16;
                        discr[y][x][ind] |= (ushort)1 << pos;
                    }
                    histBuff[y][x].empty();
                }
            }
        }
        morphologyEx(out, outOpen, MORPH_OPEN, strel);
//        imshow("input", frame);
//        imshow("output", outOpen);
//        waitKey(1);
//        sprintf(filename, "out%06d.jpg", i);
//		imwrite(path + "outputCwisarDH/" + filename, outOpen);
        sprintf(filename, "in%06d.jpg", i+1);
        frame = imread(path + filename, CV_LOAD_IMAGE_UNCHANGED);
//        cout << endl << i << endl;
    }
    long int end = getTickCount();
    int64 e = getTickCount();
    cout << path << " CwisarDH time: " << end-start << endl;
    cout << "time dword: " << e - st << endl;
    waitKey(0);
//    destroyWindow("input");
//    destroyWindow("output");


 // ZWALNIANIE PAMIĘCI - NIE TYKAC!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            delete [] discr[i][j];
        }
        delete [] discr[i];
    }
    delete [] discr;
    return 0;
}
