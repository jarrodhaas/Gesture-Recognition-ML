
//
//  BarDrawer.cpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-24.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#include "BarDrawer.hpp"

Scalar yellow() { return Scalar(0,255,255); }
Scalar blue() { return Scalar(255, 0, 0); }
Scalar green() { return Scalar(0,255,0); }
Scalar red() { return Scalar(0, 0, 255); }
Scalar magenta() { return Scalar(255, 0, 255); }
Scalar white() { return Scalar(255,255,255); }
Scalar cyan() { return Scalar(255,255,0); }

void BarDrawer::drawLine(Mat &mat, int offset, float likely) {
    assert(likely <= 1);
    Point topLeft = Point(nineties + offset * barWidth, 0 + screenHeight * (1-likely) - 50);
    Point bottomRight = Point(nineties + offset * barWidth, screenHeight - 50);
    Scalar color;
    String letter = "";
    switch (offset) {
        case 0: color = yellow(); letter = "L"; break;
        case 1: color = blue(); letter = "O"; break;
        case 2: color = green(); letter = "X"; break;
        case 3: color = red(); letter = "M"; break;
        case 4: color = white(); letter = "S"; break;
        case 5: color = cyan(); letter = "S"; break;
        default:
            break;
    }
    Point letterRectangle = Point(nineties + offset * barWidth - 5, screenHeight - 20);
    rectangle(mat, topLeft, bottomRight, color, 10);
    putText(mat, letter, letterRectangle,
            FONT_HERSHEY_COMPLEX_SMALL,
            0.6,
            white(),
            1, CV_AA);
    
}

void BarDrawer::draw(Mat &mat, VectorFloat likelihood) {
    //the bars occupy the right 1/10th of the screen.
    //there are 6 bars
    //tops and bottoms of each are relative to the 90th percentile of screenWidth
    //assert(likelihood.size() == spellsCount);
    //TODO:
    for (int i = 0; i < likelihood.size(); i++) {
        drawLine(mat, i, likelihood[i]);
    }
    
    
    
}

