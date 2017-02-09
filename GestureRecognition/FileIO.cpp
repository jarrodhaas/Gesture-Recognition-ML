//
//  Saver.cpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-21.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#include "FileIO.hpp"
#include <iostream>

using namespace GRT;
using namespace std;

void FileIO::appendGesture(VectorFloat frameFloats, Spell spell) {
    ofstream myfile;
//   MatrixFloat oldMatrix;
//    oldMatrix.load("training.csv");
//    
//    int rowCount = oldMatrix.getNumRows();
//    grt_assert( oldMatrix.getNumCols() == FRAMES_PER_GESTURE + 1);
//    
//    MatrixFloat appendee = MatrixFloat(1,frameFloats.size());
//    
//    grt_assert( appendee.getNumCols() == FRAMES_PER_GESTURE + 1);
//    
//    MatrixFloat savee = MatrixFloat(rowCount + 1, FRAMES_PER_GESTURE);
//    
//    frameFloats[0] = spell;
    
//    for (int i = 0; i < FRAMES_PER_GESTURE + 1; i++) {
//        savee[rowCount][i] = 1;
//    }
//    savee[rowCount
    
    
    ///////////////
        
    ifstream filetest("training.csv");
    
    if (filetest) {
      myfile.open ("training.csv", ios::app);
    }
    
    else {
      myfile.open ("training.csv", ios::out);
    }
    
    
    // format of 1d vector
    myfile << spell << ", ";
    for (int i = 1; i < FRAMES_PER_GESTURE * 4; i++) {
        myfile << frameFloats[i] << ",";
    }
    myfile << frameFloats[FRAMES_PER_GESTURE * 4] << endl;
    myfile.close();
    
    
}
          
void FileIO::load() {
    MatrixFloat oldMatrix;
    oldMatrix.load("training.csv");
    cout << "oldMatrix.getNumCols(): " << oldMatrix.getNumCols() << endl;
}
