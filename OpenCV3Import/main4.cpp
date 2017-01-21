#include <iostream>
#include "stdlib.h"
#include "math.h"
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


// capture the video from webcam
VideoCapture cap(0);

// object tracking threshold values
int iLowH = 170;
int iHighH = 179;

int iLowS = 150;
int iHighS = 255;

int iLowV = 60;
int iHighV = 255;


Normalizer normalizer;
GestureRecognitionPipeline pipeLineToTest;



float maximumX(VectorFloat vector, int size) {
    int offsetX = 4;
    int max = 0;
    for (int i = int(vector.size()) - offsetX; i > vector.size()-(size*4); i-=4) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max;
}


float minimumX(VectorFloat vector, int size) {
    int offsetX = 4;
    int min = 9999;
    for (int i = int(vector.size()) - offsetX; i > vector.size()-(size*4); i-=4) {
        if (vector[i] < min) {
            min = vector[i];
        }
    }
    return min;
}

float maximumY(VectorFloat vector, int size) {
    int offsetY = 3;
    int max = 0;
    for (int i = int(vector.size()) - offsetY; i > vector.size()-(size*4); i-=4) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max;
}


float minimumY(VectorFloat vector, int size) {
    int offsetY = 3;
    int min = 9999;
    for (int i = int(vector.size()) - offsetY; i > vector.size()-(size*4); i-=4) {
        if (vector[i] < min) {
            min = vector[i];
        }
    }
    return min;
}

int getWandThresh(VectorFloat frameFloats, int size) {
    
    int wandThreshX = 0;
    int wandMaxX = 0;
    int wandMinX = 0;
    int wandThreshY = 0;
    int wandMaxY = 0;
    int wandMinY = 0;

    
    wandMaxX = maximumX(frameFloats, size);
    wandMinX = minimumX(frameFloats, size);
    wandThreshX = wandMaxX-wandMinX;
    
    wandMaxY = maximumY(frameFloats, size);
    wandMinY = minimumY(frameFloats, size);
    wandThreshY = wandMaxY-wandMinY;
    
    if (wandThreshX > wandThreshY) {
        return wandThreshX;
    }
    else {
        return wandThreshY;
    }
    
}


// get center of tracked image

Moments getMoments(Mat* imgOriginal) {


    //Convert the captured frame from BGR to HSV
    Mat imgHSV;
    cvtColor(*imgOriginal, imgHSV, COLOR_BGR2HSV);

    Mat imgThresholded;
    inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //Calculate the moments of the thresholded image
    return moments(imgThresholded);

}


int captureFrame(Mat *imgOriginal) {
    
    
    bool bSuccess = cap.read(*imgOriginal);
    
    if (!bSuccess){
        cout << "Cannot read a frame from video stream" << endl;
        
        }
    
    return 1;
}


// initialize precapture window
int doCapture() {
    
    // Previous positional coordinates of center of tracked object -- needed to calculate velocity
    int iLastX = -1;
    int iLastY = -1;
    
    
    //////////////////
    /// IN CONSTRUCTION
    //////////////////
    
    // Frame buffer - Holds our tracking data for the last FRAMES_BY_GESTURE number of frames
    VectorFloat frameFloats = VectorFloat( FRAMES_PER_GESTURE*PROPERTIES_PER_FRAME );
    
    // Holds the data after it has been normalized and velocity has been stripped out (found to be detrimental to classification)
    VectorFloat noVelocity = VectorFloat(60);
    
    //Capture an image from the camera
    Mat imgOriginal;
    
    cap.read(imgOriginal);
    
    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
    
    // init counter for frames since last movement
    int frame_counter = 0;
    // sample counter counts number of successful capture training samples
    int sample_counter = 0;
    
    bool now_sampling = false;
    
    double dM01;
    double dM10;
    double dArea;
    
    int posX;
    int posY;
    
    int key_press;
    
    
    
    // timing info for velocity capture
    high_resolution_clock::time_point flopTime = high_resolution_clock::now();
    long mSec;
    
    
    FileIO filer;
    
    int baseIndex;
    
    
    // Main loop
    while (true) {
        
        
        // read a new frame from video, spawn a new thread as well
        
        //std::thread t1(capThread, std::ref(imgOriginal));

        captureFrame(&imgOriginal);
        
        
        // find center of tracked object in capture frame
        Moments oMoments = getMoments(&imgOriginal);
    
        dM01 = oMoments.m01;
        dM10 = oMoments.m10;
        dArea = oMoments.m00;
        
        
        // if the area <= 10000 assume no object in the image
        // it's because of the noise, the area is not zero
        
        if (dArea > 10000) {
            
            //calculate the position of the tracked object
            posX = dM10 / dArea;
            posY = dM01 / dArea;
            
            
            // if frame buffer is full (30 frames) then start capture sequence
            
            
            if (now_sampling == true && frame_counter < FRAMES_PER_GESTURE) {
                
                // start drawing line through coordinates of tracked object
                if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0) {
                    line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);

                    
                    //capture velocity data
                    high_resolution_clock::time_point tempTime = flopTime;
                    flopTime = high_resolution_clock::now();
                    mSec = duration_cast<milliseconds>( flopTime - tempTime ).count();
                    
                    
                    //TODO: record each frame into a gesture vector...
                    baseIndex = (frame_counter * 4);
    
                    frameFloats[baseIndex] = float(posX);
                    frameFloats[baseIndex+ 1] = float(posY);
                    frameFloats[baseIndex+ 2] = float(iLastX - posX)/float(mSec);
                    frameFloats[baseIndex+ 3] = float(iLastY - posY)/float(mSec);

                    
                    // if this is the first frame, set the velocity of the first coordinate to 0
                    if (baseIndex == 0) {
                        frameFloats[3] = 0;
                        frameFloats[4] = 0;
                    }

                }
                
                frame_counter++;
                
            
                
            // once we get 30 frames stop capturing, and now_sampling is true
            } else if (frame_counter == FRAMES_PER_GESTURE) {
                
                // make sure we stop collecting data, but keep the camera capture loop going
                frame_counter++;
                cout << "Finished sampling! :)" << endl;
                
                // offset set to one because of class label
                noVelocity = normalizer.noVelocityPredict(frameFloats);
                
                
            }
            
            // record the center of the tracked object from current frame
            iLastX = posX;
            iLastY = posY;
        }
        
        
        // add current trace of tracking data (the red line)
        imgOriginal = imgOriginal + imgLines;
        
        
        // generate mirror image, since openCV captures reverse mirror by default
        cv::flip(imgOriginal,imgOriginal,1);
    
        
        imshow("Capture Stream",imgOriginal);
        
        
        ////////
        // potentially offload keypress stuff to own function - back burner
        ////////
        
        
        key_press = waitKey(1);
        
        if (key_press == 27) {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        
        // if spacebar is hit, start capturing data
        else if (key_press == 13 && frame_counter >=30) {
            
            now_sampling = true;
            frame_counter = 0;
            
            filer.appendGesture(frameFloats, circa);
            sample_counter++;
            cout << "captured " << sample_counter << " samples. " << endl;
            imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
           
        }
        
        else if (key_press == 32) {
            now_sampling = true;
            frame_counter = 0;
            cout << "capture" << endl;
            imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
            
            
        }
            
        
    }
    
    return 0;
}



//
// Real time classification of gesture input from video feed
//



int classify() {
    
    VectorFloat frameFloats = VectorFloat((FRAMES_PER_GESTURE*4));
    VectorFloat frameFloatsReset = VectorFloat((FRAMES_PER_GESTURE*4));
    VectorFloat frameFloatsAliased = VectorFloat((FRAMES_PER_GESTURE*4));
    
    // sensitivity thresholds for classification of each spell
    float spellThresh[] = {.7, .78, .60, .60, .54};
    
    // info for displaying text on screen
    string spellNames[] = {"Line", "Circle", "Expulsio", "McDonalds", "Serpensensio"};
    int text_origin[] = {20, 30};
    int font = CV_FONT_HERSHEY_SIMPLEX;
    Scalar text_color = Scalar(255,255,255);
    
    // for debugging: track highest prediction for each class
    VectorFloat bestThresh = VectorFloat(NUM_CLASSES);
    // debugging: tracks the total classifications beyond threshold per spell
    int guessList[NUM_CLASSES];
    
    // draws prediction bars on screen
    BarDrawer Drawer;
    
    // stores the probabilities for a given classification
    VectorFloat classesFloat = VectorFloat(4);

    // previous frame's center of tracked object
    int iLastX = -1;
    int iLastY = -1;
    int frame_counter = 0;
    
    //Load the pipeline from a file
    if( !pipeLineToTest.load( "AdaBoostPipelineTest6" ) ){
        cout << "ERROR: Failed to load the pipeline!\n";
               //return EXIT_FAILURE;
    }
    
    //Capture an image from the camera
    Mat imgOriginal;
    cap.read(imgOriginal);
    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
    
    
    high_resolution_clock::time_point betweenTime = high_resolution_clock::now();
    
    int baseIndex = 0;
    int posX = -1;
    int posY = -1;
    
    
    int wandThresh = 0;
    int celebrate = 0;
    int celeTimer = 0;
    int thickness;
    
    
    double dM01;
    double dM10;
    double dArea;
   
    // init guess array
    //                                                      WE SHOULD CHANGE CLASS LABELS 6->5 (get rid of magic numbers)
    for (int i =0; i<7; i++) {
        guessList[i] = 0;
    }
    
    
    while (true) {
        
        captureFrame(&imgOriginal);
        
        //measure time of execution...
        high_resolution_clock::time_point time1 = high_resolution_clock::now();

        // process image and calculate center of tracked object
        Moments oMoments = getMoments(&imgOriginal);
        
        dM01 = oMoments.m01;
        dM10 = oMoments.m10;
        dArea = oMoments.m00;
        
        
        // if the area <= 10000 assume no object in the image
        // it's because of the noise, the area is not zero
        if (dArea > 10000) {
            
            //calculate the position of the ball
            posX = (dM10 / dArea);
            posY = (dM01 / dArea);
            
            // calculate time passed between loops and call this the time/frame
            high_resolution_clock::time_point beforeT = betweenTime;
            high_resolution_clock::time_point betweenTime = high_resolution_clock::now();
            long mSec = duration_cast<milliseconds>( beforeT - betweenTime ).count();
            float sec = float(mSec)*1000;
            
            // has the wand moved in the last FRAME_THRESHOLD frames? only try to classify if this is true
            wandThresh = getWandThresh(frameFloats, FRAME_THRESHOLD);
            
            
            
            //
            //
            // FIX THIS TO MAKE SURE 29th frame at index 28 has a frame on the first 60..
            //
            //
            // if less than FRAMES_PER_GESTURE-1 frames, add frame
            if (frame_counter < FRAMES_PER_GESTURE - 1) {
            
                
                // store current frame data
                frameFloats[baseIndex] = roundf(float(posX));
                frameFloats[baseIndex+ 1] = roundf(float(posY) );
                frameFloats[baseIndex+ 2] = roundf((float(iLastX - posX)))/float(sec);
                frameFloats[baseIndex+ 3] = roundf((float(iLastY - posY)))/float(sec);
                
                baseIndex += 4;
                frame_counter ++;
                
                if (iLastX == -1 && iLastY == -1) {
                    frameFloats[2] = 0;
                    frameFloats[3] = 0;
                }
    
            }
            // If we have enough frames, we just want to keep the last thirty and potentially make a guess thereon
            else {
                
                // move last 30 frames back 1 frame
                for (int i=4; i < (FRAMES_PER_GESTURE * 4); i+=4) {
                    frameFloats[i-4] = frameFloats[i];
                    frameFloats[i-3] = frameFloats[i+1];
                    frameFloats[i-2] = frameFloats[i+2];
                    frameFloats[i-1] = frameFloats[i+3];
                }
                
                // add the new frame to end of set
                frameFloats[baseIndex] = roundf(float(posX));
                frameFloats[baseIndex+1] = roundf(float(posY));
                frameFloats[baseIndex+2] = roundf((float(iLastX - posX)))/float(sec);
                frameFloats[baseIndex+3] = roundf((float(iLastY - posY)))/float(sec);
                
                // if the wand has moved more than MOVEMENT_THRESHOLD we allow the classifier to take a guess
                if (wandThresh > MOVEMENT_THRESHOLD) {
                
                    // We decided to remove velocity in the interim while we explore its contribution
                    VectorFloat noVelocity = normalizer.noVelocityPredict(frameFloats);
                    
                    // Send the feature vector to the classifier
                    pipeLineToTest.predict(noVelocity);
                    
                    // Debugging: Request the highest probability label
                    unsigned int predictedClassLabel = pipeLineToTest.getPredictedClassLabel();
                    cout << "Class label: " << predictedClassLabel << endl;
                    
                    // Request the list of probabilities related to each class label on the last guess.
                    // This is useful to do some thresholding on our own
                    classesFloat = pipeLineToTest.getClassLikelihoods();
                    
                    // Debugging: Record best acheived likelihood for each class in the current videostream.
                    // This is used for manual threshold tuning
                    for (int i = 0; i < classesFloat.size(); i++) {
                        //cout << "classesFloat[" << i << "]: " << classesFloat[i] << endl;
                        
                        if (classesFloat[i] > bestThresh[i]) {
                            bestThresh[i] = classesFloat[i];
                        }
                        
                        cout << "Best Threshold[" << i << "]: " << bestThresh[i] << endl;
                    }
                    
                    // Compares prediction for each label against our manual threshold for each label
                    for (int i = 0; i < classesFloat.size(); i++) {
                        
                        if (classesFloat[i] > spellThresh[i]) {
                            // Debugging: Record number of classifications per spell -> used to build a confusion matrix
                            guessList[i]++;
                            // Triggers that user should be notified of a recognized gesture
                            celebrate = i+1;
                            // This ensures that the first valid guess is not overwritten by another
                            break;
                            
                        }
                        
                    }
                }
             
            }
            
            
            // Clear old tracking lines
            imgLines.release();
            imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
            
              // why first 30 frames same thickness?
            
              for (int i = 1; i < FRAMES_PER_GESTURE; i++) {
                
                  
                if ( frameFloats[(i-1)*4] > 0 && frameFloats[i*4] > 0 ) {
                    
                    thickness = int( pow((FRAMES_PER_GESTURE / float(FRAMES_PER_GESTURE - i + 1)),.5) * 5);
                    cout << "Thickness: " << thickness << endl;
                    //Draw a red line from the previous point to the current point
                    line(imgLines, Point(frameFloats[(i-1)*4], frameFloats[(i-1)*4+1]),
                         Point(frameFloats[i*4], frameFloats[i*4+1]), Scalar(0,0,255), thickness);
                }
            
              }
            
            
            iLastX = posX;
            iLastY = posY;
        }
        
        
        
        
        // add lines to image
        imgOriginal = imgOriginal + imgLines;
        
        // generate mirror image
        cv::flip(imgOriginal,imgOriginal,1);
        
        if(celebrate && celeTimer < 50){
            string message = "You're a wizard! You cast " + spellNames[celebrate-1] + "! Way to go.";
            //cout << message << endl;
            putText(imgOriginal, message, Point(text_origin[1], text_origin[1]), font, 1, text_color, 3);
            celeTimer += 1;
            if (celeTimer == 50){
                celebrate = 0;
                celeTimer = 0;
            }
        }
        
        
        Drawer.draw(imgOriginal , classesFloat);
        
        //imshow("Thresholded Image", imgThresholded); //show the thresholded image
        
        imshow("flipped",imgOriginal);
    
      
        
        int key = waitKey(1);
        
        if (key == 27) {
            cout << "esc key is pressed by user" << endl;
            cout << "results: ";
            
            int guessSum = 0;
            
            for (int i = 0; i < 7; i++) {
                cout << guessList[i] << ", ";
                guessSum+=guessList[i];
            }
            
            double percent = 0;
            
            cout << endl;
            
            for (int i = 0; i < 7; i++) {
                
                percent = double(guessList[i])/guessSum;
                cout << "percent class: " << i << " " << percent << endl;
            
                
            }
            
            
            break;
        }
        else if (key == 32) {
            imgLines.release();
            imgLines = Mat::zeros( imgOriginal.size(), CV_8UC3 );
            cout << "lines eliminate" << endl;
            
            
        }
        
    }
    
    //cout << "Celebrate: " << celebrate << endl;
    return celebrate;
    
 
}

///TRAINER
int main( int argc, char** argv ) {
    
    if ( !cap.isOpened() ) {
        cout << "Cannot open the web cam" << endl;
        return -1;
    }
    
 
    classify();
   
    
    return 0;
}
