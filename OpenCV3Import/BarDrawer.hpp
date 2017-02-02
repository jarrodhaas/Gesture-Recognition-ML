//
//  BarDrawer.hpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-24.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#ifndef BarDrawer_hpp
#define BarDrawer_hpp

#include "Constants.hpp"
#include <stdio.h>
#include "GRT/GRT.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace GRT;
using namespace cv;
using namespace std;

class BarDrawer {
  
  void drawLine(Mat &mat, int offset, float likely);
  
public:
  int screenHeight = 720;
  int screenWidth = 1280;
  
  int nineties = screenWidth * 0.91;
  int barWidth = 20;

  int spellsCount = 6;
  
  void draw(Mat &mat, VectorFloat likelihood);
  
};

#endif /* BarDrawer_hpp */
