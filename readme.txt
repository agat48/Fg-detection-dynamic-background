---------------------------------------------------------------------------
PBAS - Version 0.1
---------------------------------------------------------------------------

1. INTRODUCTION - PBAS
2. REQUIREMENTS
3. HOW TO USE
4. FUTURE WORK

---------------------------------------------------------------------------
1. INTRODUCTION - PBAS
---------------------------------------------------------------------------
PBAS (Pixel-based adaptive Segmenter) is a pixel-wise change detection algorithm
for video streams of different sizes, quality and frame rates. It first was presented in the 
paper:

M. Hofmann, P. Tiefenbacher, G. Rigoll 
"Background Segmentation with Feedback: The Pixel-Based Adaptive Segmenter", 
in proc of IEEE Workshop on Change Detection, 2012

We use a nonparametric background model, which consists of recently observed pixel 
and magnitude values. Our approach compares this model to the actual pixel features and then decides, 
whether its part of the foreground or background. Regarding gradual changes in the background, we apply a 
random and conservative update policy, which is executed by a adapting per-pixel-wise update rate. 
Furthermore the per-pixel-wise decision threshold for the foreground classification is controlled through 
a new idea of describing the background dynamics and leads therefore to a very sensitive foreground segmenter 
in low dynamic background areas, by handling on the other side high dynamic areas very well, also. 
Thus there are two controlling loops for each pixel adapting to the current video properties. 

So the best way to understand the behaviour and the idea behind the PBAS is to read this paper.
Furthermore there exist some informtion on the homepage of the PBAS:
https://sites.google.com/site/pbassegmenter/

---------------------------------------------------------------------------
2. REQUIREMENTS
---------------------------------------------------------------------------
- OpenCV C++ (2.1 or newer)
- C++ environment
- Editor to compile C++ Code (Visual Studio / Eclipse)

---------------------------------------------------------------------------
3. HOW TO USE
---------------------------------------------------------------------------
After you have realised a working opencv implementation, just include the pbas.h / pbas.cpp file to your project.
At the moment PBAS only supports 1- or 3-channel uchar RGB images as input. All other formats will fail.

A basic usage example could look like this:

 //Somewhere during initalization:
 #include "PBAS.h"
 #include <opencv2/opencv.hpp>

 PBAS pbas;

//you might want to change some parameters of the PBAS here...
// ....

//repeat for each frame
//Recommended step in the most cases: 
//make gaussian blur for reducing image noise
cv::Mat bluredImage;
cv::Mat pbastResult;
cv::GaussianBlur(singleFrame, bluredImage, cv::Size(5,5), 1.5);
 
 //process image and receive segmentation in pbasResult
pbas.process(&bluredImage, &pbasResult);

//Recommended step in the most cases: 
//make medianBlur on the result to reduce "salt and pepper noise"
//of the per pixel wise segmentation
cv::medianBlur(pbasResult, pbasResult, 5);


---------------------------------------------------------------------------
4. FUTURE WORK
---------------------------------------------------------------------------
- Support more formats
- mex-file for Matlab support
- ...