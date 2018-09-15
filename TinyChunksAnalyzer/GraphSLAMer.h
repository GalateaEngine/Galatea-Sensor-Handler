#pragma once
#include "stdafx.h"

class SLAMPoint
{
	Point3d location;
	double colour;
};

class GraphSLAMer
{
public:
	class SE3;
	//The main function for LS Graph SLAM. Takes input in the form of camera frames, and returns a matrix with the approximate position of the camera. 
	//Also builds a map behind the scenes for which the point cloud can be accessed by the helper functions
	//enhanced implementation of https://groups.csail.mit.edu/rrg/papers/greene_icra16.pdf
	//K: is a 3x3 real mat with the camera parameters
	//pi: perspective projection function
	SE3 LS_Graph_SLAM(Mat cameraFrame);

	//Sets up matrices and other things
	void Initialize_LS_Graph_SLAM();

	std::list<SLAMPoint> get3dPoints();
};