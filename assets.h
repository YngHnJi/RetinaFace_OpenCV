#include <vector>
#include <string>

struct Bbox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    float ppoint[10]; //(x,y) left eye, right eye, nose tip, left angle of mouth, right angle of mouth
    float regreCoord[4];
};