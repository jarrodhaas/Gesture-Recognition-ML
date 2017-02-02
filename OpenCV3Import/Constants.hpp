//
//  Constants.hpp
//  OpenCV3Import
//
//  Created by June Kim on 2016-11-21.
//  Copyright © 2016 Soulcast-team. All rights reserved.
//

#ifndef Constants_hpp
#define Constants_hpp

#include <stdio.h>



const int FRAMES_PER_GESTURE = 30;
const int PAGE_BUFFER_SIZE = 1000;
const int ALIAS_FACTOR = 10;
const int NUM_CLASSES = 5;
const int FRAME_THRESHOLD = 3;
const int MOVEMENT_THRESHOLD = 50;


struct Celebrate {
    int counter = 0;
    int timer = 0;
};


enum Spell: int {
    leviosa = 1,
    circa = 2,
    expulsio = 3,
    mophiosa = 5,
    serpincio = 6
    
};


#endif /* Constants_hpp */
