#include <iostream>
#include "stdlib.h"
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "GRT/GRT.h"

#include <thread>
#include <chrono>

#include "Normalizer.hpp"
#include "XYVV.hpp"
#include "SpellEnum.hpp"
#include "FileIO.hpp"
#include "Constants.hpp"
#include "BarDrawer.hpp"

using namespace cv;
using namespace GRT;
using namespace std;
using namespace std::chrono;


// https://github.com/nickgillian/grt/blob/master/examples/ClassificationModulesExamples/SVMExample/SVMExample.cpp


//capture the video from webcam
VideoCapture cap(0);

// hsv control window initial values
int iLowH = 170;
int iHighH = 179;

int iLowS = 150;
int iHighS = 255;

int iLowV = 60;
int iHighV = 255;

/*int iLowH = 30;
int iHighH = 86;

int iLowS = 30;
int iHighS = 235;

int iLowV = 140;
int iHighV = 230;
*/


Normalizer normalizer;

void capThread(Mat &imgOriginal) {
    
    //Mat imgOriginal;
    
    bool bSuccess = cap.read(imgOriginal);
    
    if (!bSuccess){
        cout << "Cannot read a frame from video stream" << endl;
        
    }
}

void capThread2(Mat &imgOriginal) {
    
    //Mat imgOriginal;
    
    bool bSuccess = cap.read(imgOriginal);
    
    if (!bSuccess){
        cout << "Cannot read a frame from video stream" << endl;
        
    }
    
}

float maximumX(VectorFloat vector, int size) {
    int max = 0;
    for (int i = vector.size() - 4; i > vector.size()-(size*4); i-=4) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max;
}


float minimumX(VectorFloat vector, int size) {
    int min = 9999;
    for (int i = vector.size() - 4; i > vector.size()-(size*4); i-=4) {
        if (vector[i] < min) {
            min = vector[i];
        }
    }
    return min;
}

float maximumY(VectorFloat vector, int size) {
    int max = 0;
    for (int i = vector.size() - 3; i > vector.size()-(size*4); i-=4) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max;
}


float minimumY(VectorFloat vector, int size) {
    int min = 9999;
    for (int i = vector.size() - 3; i > vector.size()-(size*4); i-=4) {
        if (vector[i] < min) {
            min = vector[i];
        }
    }
    return min;
}

// initialize precapture window
int doCapture() {
    
    int iLastX = -1;
    int iLastY = -1;
    
    VectorFloat frameFloats = VectorFloat(FRAMES_PER_GESTURE*4 + 1);
    VectorFloat noVelocity = VectorFloat(60);
    
    Mat normTest;
    
    //Capture a temporary image from the camera
    Mat imgTmp;
    
    cap.read(imgTmp);
    
    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;
    
    // init counter for frames since last movement
    int frame_counter = 0;
    int sample_counter = 0;
    
    Mat imgOriginal;
    imgOriginal = imgTmp;
    
    bool now_sampling = false;
    
    high_resolution_clock::time_point flopTime = high_resolution_clock::now();
    
    
    FileIO filer;
    int numCaptured = 0;
    
    while (true) {
        
        //TODO: blocking function. spawn thread.
        // read a new frame from video.
        
        flopTime = high_resolution_clock::now();
        
        std::thread t1(capThread, std::ref(imgOriginal));

        
        
        //Convert the captured frame from BGR to HSV
        Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
        
        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
        
        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        
        //morphological closing (removes small holes from the foreground)
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        
        //Calculate the moments of the thresholded image
        Moments oMoments = moments(imgThresholded);
        
        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;
        
        // if the area <= 10000 assume no object in the image
        // it's because of the noise, the area is not zero
        if (dArea > 10000) {
            //calculate the position of the ball
            int posX = dM10 / dArea;
            int posY = dM01 / dArea;
            
//            flopTime = high_resolution_clock::now();
            
            // if < 30 for x frames, then start capture sequence
            
            
            if (now_sampling == true && frame_counter < FRAMES_PER_GESTURE) {
//                 cout << "FRAME CAPTURE INITIATED" << endl;
                
                
                if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
                    line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);

                    //TODO: test boundary times...
                    high_resolution_clock::time_point tempTime = flopTime;
                    flopTime = high_resolution_clock::now();
                    long mSec = duration_cast<milliseconds>( flopTime - tempTime ).count();
                    if (mSec == 0 || mSec > 100) {
                        cout << "frame_counter: " << frame_counter << endl;
                    }
                    //TODO: record each frame into a gesture vector...
                    int baseIndex = (frame_counter * 4)+1;
    
                    frameFloats[baseIndex] = float(posX);
                    frameFloats[baseIndex+ 1] = float(posY);
                    frameFloats[baseIndex+ 2] = float(iLastX - posX)/float(mSec);
                    frameFloats[baseIndex+ 3] = float(iLastY - posY)/float(mSec);

                    
                    if (baseIndex-1 == 0) {
                        frameFloats[3] = 0;
                        frameFloats[4] = 0;
                    }

                }
                
                frame_counter++;
            } else if (frame_counter == FRAMES_PER_GESTURE) {
                
                sample_counter++;
                frame_counter++;
                cout << "Finished sampling! :)" << endl;
                
                // offset set to one bc of class label
                noVelocity = normalizer.noVelocityPredict(frameFloats,1);
                
                // multiply normed values by 20
                for (int i =0; i < 60; i++) {
                    
                    noVelocity[i] = noVelocity[i]*100;
                    
                }
                
                
                // draw rescaled normed values for testing, should display in upper right
                for (int i = 1; i < 30; i++) {
                    line(normTest, Point(noVelocity[i*2], noVelocity[i*2+1]),
                         Point(noVelocity[(i-1)*2], noVelocity[(i-1)*2+1]), Scalar(0,0,255), 2);
                    
                }

                
                
            }
            
            
            iLastX = posX;
            iLastY = posY;
        }
        
        
        imgOriginal = imgOriginal + normTest;
        
        // add lines to image
         imgOriginal = imgOriginal + imgLines;
        
        // generate mirror image
        cv::Mat dst;
        cv::flip(imgOriginal,dst,0);
        Point2f src_center(dst.cols/2.0F, dst.rows/2.0F);
        cv::Mat rot_matrix = getRotationMatrix2D(src_center, 180.0, 1.0);
        cv::Mat rotated_img(Size(dst.size().height, dst.size().width), dst.type());
        warpAffine(dst, rotated_img, rot_matrix, dst.size());
        
        
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        
        imshow("flipped",rotated_img);
        
        t1.join();
        
        
        int key_press = waitKey(1);
        
        if (key_press == 27) {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        else if (key_press == 13 && frame_counter >=30) {
            now_sampling = true;
            frame_counter = 0;
            
            filer.appendGesture(frameFloats, circa);
            numCaptured++;
            cout << "captured " << numCaptured << endl;
            imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
            normTest = Mat::zeros( imgTmp.size(), CV_8UC3 );
            
        }
        
        else if (key_press == 32) {
            now_sampling = true;
            frame_counter = 0;
            cout << "capture" << endl;
            imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
            normTest = Mat::zeros( imgTmp.size(), CV_8UC3 );
            
        }
            
        
    }
    
    return 0;
}

GestureRecognitionPipeline pipeLineToTest;


////the real time part...
int doMagic() {
    
    VectorFloat frameFloats = VectorFloat((FRAMES_PER_GESTURE*4));
    VectorFloat frameFloatsReset = VectorFloat((FRAMES_PER_GESTURE*4));
    

    // ver 5 float spellThresh[] = {.7, .78, .60, .63, .65};
    //float spellThresh[] = {.7, .78, .60, .60, .54}; -- adaboosttest6
 
    
    float spellThresh[] = {.7, .82, .49, .63, .5};
    
    string spellNames[] = {"line", "circle", "expulsio", "McDonalds", "Serpensensio"};
    int location[] = {20, 30};
    int font = CV_FONT_HERSHEY_SIMPLEX;
    
    Scalar color = Scalar(255,255,255);
    
    // track the best threshold for each category
    VectorFloat bestThresh = VectorFloat(5);
    
    int resultArr[1000];
    int resultCount = 0;
    
    
    int iLastX = -1;
    int iLastY = -1;
    
    int frame_counter = 0;
    
    // load the training data
    
    //Load the pipeline from a file
    if( !pipeLineToTest.load( "AdaBoostCrossTest" ) ){
        cout << "ERROR: Failed to load the pipeline!\n";
               //return EXIT_FAILURE;
    }
    
    //Capture a temporary image from the camera
    Mat imgTmp;
    cap.read(imgTmp);
    
    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
    

    high_resolution_clock::time_point timex = high_resolution_clock::now();
    high_resolution_clock::time_point time2;
    long totalDuration;
    float frameRate;
    
    Mat imgOriginal;
    capThread(imgOriginal);
    
    high_resolution_clock::time_point betweenTime;
    int baseIndex = 0;
    int posX = -1;
    int posY = -1;
    
    int wandThreshX = 0;
    int wandMaxX = 0;
    int wandMinX = 0;
    int wandThreshY = 0;
    int wandMaxY = 0;
    int wandMinY = 0;
    int celebrate = 0;
    int celeTimer = 0;
    
    
    
    BarDrawer Drawer;
    VectorFloat classesFloat = VectorFloat(4);
    
    while (true) {

        high_resolution_clock::time_point time0 = high_resolution_clock::now();
        
        std::thread t1(capThread, std::ref(imgOriginal));
        
        //measure time of execution...
        high_resolution_clock::time_point time1 = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>( time1 - time0 ).count();
        //cout << "Milliseconds to capture: " << duration << endl;
        //
        
        //Convert the captured frame from BGR to HSV
        Mat imgHSV;
        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
        
        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image
        
        //morphological opening (removes small objects from the foreground)
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        
        //morphological closing (removes small holes from the foreground)
        dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
        
        //Calculate the moments of the thresholded image
        Moments oMoments = moments(imgThresholded);
        
        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;
        
        
        
        // if the area <= 10000 assume no object in the image
        // it's because of the noise, the area is not zero
        if (dArea > 10000) {
            
            //calculate the position of the ball
            posX = (dM10 / dArea);
            posY = (dM01 / dArea);
            
         
            high_resolution_clock::time_point beforeT = betweenTime;
            high_resolution_clock::time_point betweenTime = high_resolution_clock::now();
            long mSec = duration_cast<milliseconds>( beforeT - betweenTime ).count();;
            float sec = float(mSec)*1000;
            
            // has the wand moved beyond threshold?
            
            wandMaxX = maximumX(frameFloats, 3);
            wandMinX = minimumX(frameFloats, 3);
            wandThreshX = wandMaxX-wandMinX;
            
            wandMaxY = maximumY(frameFloats, 3);
            wandMinY = minimumY(frameFloats, 3);
            wandThreshY = wandMaxY-wandMinY;
            
            // if less than FRAMES_PER_GESTURE-1 frames, add frame
            
            if (frame_counter < FRAMES_PER_GESTURE - 1) {
            
                
              // store current frame data
            
                frameFloats[baseIndex] = roundf(float(posX) / ALIAS_FACTOR);
                frameFloats[baseIndex+ 1] = roundf(float(posY) / ALIAS_FACTOR);
                frameFloats[baseIndex+ 2] = roundf((float(iLastX - posX)/ALIAS_FACTOR))/float(sec);
                frameFloats[baseIndex+ 3] = roundf((float(iLastY - posY)/ALIAS_FACTOR))/float(sec);

                
                
                
                baseIndex += 4;
                
              if (iLastX == -1 && iLastY == -1) {
                  frameFloats[0] = 0;
                  frameFloats[1] = 0;
              }
             
                for (int i = 0; i < 120; i++) {
                    
                    cout << frameFloats[i] << ", ";
                }
                cout << endl;
             
                
    
            }
            else if (frame_counter >= FRAMES_PER_GESTURE - 1){
                
                // move last 30 frames back 1 frame
                for (int i=4; i < (FRAMES_PER_GESTURE * 4); i+=4) {
                    frameFloats[i-4] = frameFloats[i];
                    frameFloats[i-3] = frameFloats[i+1];
                    frameFloats[i-2] = frameFloats[i+2];
                    frameFloats[i-1] = frameFloats[i+3];
                }
                
                // add the new frame to end of set
                frameFloats[baseIndex] = roundf(float(posX) / ALIAS_FACTOR);
                frameFloats[baseIndex+1] = roundf(float(posY) / ALIAS_FACTOR);
                frameFloats[baseIndex+2] = roundf((float(iLastX - posX)/ALIAS_FACTOR))/float(sec);
                frameFloats[baseIndex+3] = roundf((float(iLastY - posY)/ALIAS_FACTOR))/float(sec);
                
                
                if (wandThreshX > 50 || wandThreshY > 50) {
                
                    // normalize data
                    //VectorFloat normedPredictable = normalizer.normalizePredict(frameFloats);
                    VectorFloat noVelocity = normalizer.noVelocityPredict(frameFloats,0);
                    
                    
    //                for (int i = 0; i < 120; i++) {
    //                 cout << normedPredictable[i] << ", ";
    //                 }
                    
                    // send to pipeline
                
                    ////// change this as well!
                    
                    
                    pipeLineToTest.predict(noVelocity );
                    unsigned int predictedClassLabel = pipeLineToTest.getPredictedClassLabel();
                    classesFloat = pipeLineToTest.getClassLikelihoods();
                    for (int i = 0; i < classesFloat.size(); i++) {
                        cout << "classesFloat[" << i << "]: " << classesFloat[i] << endl;
                        
                        if (classesFloat[i] > bestThresh[i]) {
                            bestThresh[i] = classesFloat[i];
                        }
                        
                        cout << "Best Threshold[" << i << "]: " << bestThresh[i] << endl;
                    }
                    
                    for (int i = 0; i < classesFloat.size(); i++) {
                        
                        if (classesFloat[i] > spellThresh[i]) {
                           
                            celebrate = i+1;
                            break;
                            
                        }
                        
                    }
                }
             
            }
            
            if (frame_counter < FRAMES_PER_GESTURE - 1) {
              frame_counter++;
            }
            
            // draw magic lines
        
            
            imgLines.release();
            imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
            
         
              for (int i = 1; i < FRAMES_PER_GESTURE; i++) {
                
                if ( frameFloats[(i-1)*4] > 0 && frameFloats[i*4] > 0 ) {
                    //Draw a red line from the previous point to the current point
                    line(imgLines, Point(frameFloats[(i-1)*4], frameFloats[(i-1)*4+1]),
                         Point(frameFloats[i*4], frameFloats[i*4+1]), Scalar(0,0,255), 2);
                }
            
              }
            
            
            
                
            iLastX = posX;
            iLastY = posY;
        }
        
        
        
        
        // add lines to image
        imgOriginal = imgOriginal + imgLines;
        
        // generate mirror image
        cv::Mat dst;
        cv::flip(imgOriginal,dst,0);
        Point2f src_center(dst.cols/2.0F, dst.rows/2.0F);
        cv::Mat rot_matrix = getRotationMatrix2D(src_center, 180.0, 1.0);
        cv::Mat rotated_img(Size(dst.size().height, dst.size().width), dst.type());
        warpAffine(dst, rotated_img, rot_matrix, dst.size());
        
        if(celebrate && celeTimer < 50){
            string message = "You're a wizard! You cast " + spellNames[celebrate-1] + "! Way to go.";
            cout << message << endl;
            putText(rotated_img, message, Point(location[1], location[1]), font, 1, color, 3);
            celeTimer += 1;
            if (celeTimer == 50){
                celebrate = 0;
                celeTimer = 0;
            }
        }
        
        
        Drawer.draw(rotated_img , classesFloat);
        
        imshow("Thresholded Image", imgThresholded); //show the thresholded image
        
        imshow("flipped",rotated_img);
        

        //measuring time...
        time2 = high_resolution_clock::now();
        auto duration2 = duration_cast<milliseconds>( time2 - time1 ).count();
        //cout << "Milliseconds to calculate stuff after capturing: " << duration2 << endl;
        
        // THREAD
        high_resolution_clock::time_point time3 = high_resolution_clock::now();
        t1.join();
        high_resolution_clock::time_point time4 = high_resolution_clock::now();
        auto duration4 = duration_cast<milliseconds>( time4 - time3 ).count();
        //cout << "join time: " << duration4 << endl;
        
        
        totalDuration = duration_cast<seconds>( time2 - timex ).count();
        frameRate = float(frame_counter) / float(totalDuration);
        //cout << "average frame rate: " << frameRate << endl;
        
        int key = waitKey(1);
        
        if (key == 27) {
            cout << "esc key is pressed by user" << endl;
            cout << "results: ";
            for (int i = 0; i < resultCount; i++) {
                cout << resultArr[i] << ", ";
            }
            cout << endl << resultCount << endl;
            break;
        }
        else if (key == 32) {
            imgLines.release();
            imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
            cout << "lines eliminate" << endl;
            
            resultCount = 0;
            
            for (int i = 0; i < 100; i++) {
                resultArr[i] = 9;
            }
            
        }
        
    }
    
    cout << "Celebrate: " << celebrate << endl;
    return celebrate;
    
 
}

///TRAINER
int main( int argc, char** argv ) {
    
    if ( !cap.isOpened() ) {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }
    
    //create a window called "Control"
    namedWindow("Control", CV_WINDOW_AUTOSIZE);
    
    /*//Create trackbars in "Control" window
    //Hue (0 - 179)
    createTrackbar("LowH", "Control", &iLowH, 179);
    createTrackbar("HighH", "Control", &iHighH, 179);
    //Saturation (0 - 255)
    createTrackbar("LowS", "Control", &iLowS, 255);
    createTrackbar("HighS", "Control", &iHighS, 255);
    //Value (0 - 255)
    createTrackbar("LowV", "Control", &iLowV, 255);
    createTrackbar("HighV", "Control", &iHighV, 255);*/
    
    
    //Hue (0 - 179)
    createTrackbar("LowH", "Control", &iLowH, 179);
    createTrackbar("HighH", "Control", &iHighH, 179);
    //Saturation (0 - 255)
    createTrackbar("LowS", "Control", &iLowS, 255);
    createTrackbar("HighS", "Control", &iHighS, 255);
    //Value (0 - 255)
    createTrackbar("LowV", "Control", &iLowV, 255);
    createTrackbar("HighV", "Control", &iHighV, 255);
    
    
 
    doMagic();
   
    
    return 0;
}
