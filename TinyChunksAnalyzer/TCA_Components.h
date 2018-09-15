#pragma once
#include "stdafx.h"
#include <map>
#include <set>
#include <list>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

struct Line
{
	cv::Point start;
	cv::Point end;
	float colourDif;
};


class TCA_Object
{
public:

	int numpoints;
	int numobjects;
	vector<cv::Point> points;
	list<Line> lines;
	float relative_direction;
	float relative_speed;

	//bounding box
	float x;
	float y;
	float width;
	float height;
};


class TCA_Video
{
public:
	int objects;
	map<int, TCA_Object> objectMap;
	set<cv::Point> pSet;
	cv::Mat lastFrame;

	float colourFudgeMax = 1.8;
	float colourFudgeMin = 0.2;

	//openCV stuff
	cv::VideoCapture cap;

	//states
	bool ready;
	bool useNaiveEdge = false;
	bool useGraphSLAM = true;

	//SLAM runners
	GraphSLAMer gs;

	TCA_Video();

	~TCA_Video();

	cv::Mat calcFrameDifference(cv::Mat oldFrame, cv::Mat newFrame);

	bool findNearbyPoint(cv::Point start, int width, int height, set<cv::Point>& set, cv::Point& hit);

	//object finder with naive edge finder
	bool processFrameData(cv::Mat data);



	bool update();

};

class TCA_Audio
{

};

class TCA_Text
{

};