//
//  Normalizer.cpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-11.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#include "Normalizer.hpp"

/// normalizes a vector of size Frames/gesture * Properties/frame + 1
/// the target is at index 0.
void Normalizer::normalize(VectorFloat &frames) {
    }


VectorFloat Normalizer::noVelocityPredict(VectorFloat frames) {
    int framesPerGesture = int(frames.size()) / PROPERTIES_PER_FRAME;
    // load x's and y's, into an array to calculate min/max
    VectorFloat xPositions = VectorFloat(framesPerGesture);
    VectorFloat yPositions = VectorFloat(framesPerGesture);
    
    // make a new frames array to return, of length 60
    VectorFloat newFrames = VectorFloat(framesPerGesture*2);
    
    for (int i = 0; i < framesPerGesture; i++) {
        int xIndex = i * PROPERTIES_PER_FRAME;
        xPositions[i] = frames[xIndex];
        int yIndex = xIndex + 1;
        yPositions[i] = frames[yIndex];
    }
    // calculate min, max, spread
    float minX = minimum(xPositions);
    float maxX = maximum(xPositions);
    float spreadX = maxX - minX;
    float minY = minimum(yPositions);
    float maxY = maximum(yPositions);
    float spreadY = maxY - minY;
    float maxSpread = spreadX > spreadY ? spreadX : spreadY;
    
    std::cout << "minX: "<<minX<< " maxX: " <<maxX<< " minY: " <<minY<< " maxY: " <<maxY<< std::endl;
    
    // normalize x, y, velX, velY
    for (int i = 0; i < framesPerGesture; i++) {
        int xIndex = i * 2;
        int yIndex = xIndex + 1;
       
        newFrames[xIndex] = (xPositions[i] - minX)/maxSpread;
        newFrames[yIndex] = (yPositions[i] - minY)/maxSpread;
        
    }

    
  return newFrames;
    
}


VectorFloat Normalizer::normalizePredict(VectorFloat frames) {
    int framesPerGesture = int(frames.size()) / PROPERTIES_PER_FRAME;
    // load x's and y's, into an array to calculate min/max
    VectorFloat xPositions = VectorFloat(framesPerGesture);
    VectorFloat yPositions = VectorFloat(framesPerGesture);
    for (int i = 0; i < framesPerGesture; i++) {
        int xIndex = i * PROPERTIES_PER_FRAME;
        xPositions[i] = frames[xIndex];
        int yIndex = xIndex + 1;
        yPositions[i] = frames[yIndex];
    }
    // calculate min, max, spread
    float minX = minimum(xPositions);
    float maxX = maximum(xPositions);
    float spreadX = maxX - minX;
    float minY = minimum(yPositions);
    float maxY = maximum(yPositions);
    float spreadY = maxY - minY;
    float maxSpread = spreadX > spreadY ? spreadX : spreadY;
    
    std::cout << "minX: "<<minX<< " maxX: " <<maxX<< " minY: " <<minY<< " maxY: " <<maxY<< std::endl;
    
    // normalize x, y, velX, velY
    for (int i = 0; i < framesPerGesture; i++) {
        int xIndex = i * PROPERTIES_PER_FRAME;
        int yIndex = xIndex + 1;
        int velXIndex = yIndex + 1;
        int velYIndex = velXIndex + 1;
        frames[xIndex] = (xPositions[i] - minX)/maxSpread;
        frames[yIndex] = (yPositions[i] - minY)/maxSpread;
        frames[velXIndex] = frames[velXIndex] / maxSpread;
        frames[velXIndex] = frames[velYIndex] / maxSpread;
    }
    return frames;
}

float Normalizer::minimum(VectorFloat vector) {
    int min = 9999;
    for (int i = 0; i < vector.size(); i++) {
        if (vector[i] < min) {
            min = vector[i];
        }
    }
    return min;
}

float Normalizer::maximum(VectorFloat vector) {
    int max = 0;
    for (int i = 0; i < vector.size(); i++) {
        if (vector[i] > max) {
            max = vector[i];
        }
    }
    return max;
}
