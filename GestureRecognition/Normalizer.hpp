//
//  Normalizer.hpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-11.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#ifndef Normalizer_hpp
#define Normalizer_hpp

#include <stdio.h>
#include "GRT/GRT.h"

using namespace GRT;

/*
 USAGE:
 
 Normalizer normalizer;
 
 loop:
 int floatsIndexBase = frameCounter * PROPERTIES_PER_FRAME;
 frameFloats[floatsIndexBase + 1] = frame.posX;
 frameFloats[floatsIndexBase + 2] = frame.posY;
 frameFloats[floatsIndexBase + 3] = frame.velX;
 frameFloats[floatsIndexBase + 4] = frame.velY;
 :end
 
 normalizer.normalize(frameFloats);
 
 
 */
const int PROPERTIES_PER_FRAME = 4;

class Normalizer {
    
public:
    
    void normalize(MatrixFloat &frames);
    void normalize(VectorFloat &frames);
    VectorFloat normalizePredict(VectorFloat frames);
    VectorFloat noVelocityPredict(VectorFloat frames);
    
private:
    float minimum(VectorFloat vector);
    float maximum(VectorFloat vector);
};





#endif /* Normalizer_hpp */
