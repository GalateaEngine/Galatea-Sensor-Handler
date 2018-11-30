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
	cv::Point2f start;
	cv::Point2f end;
	float colourDif;
};


class TCA_Object
{
public:

	int numpoints;
	int numobjects;
	vector<cv::Point2f> points;
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
	set<cv::Point2f> pSet;
	cv::viz::Viz3d viewer;
	bool widgetSet = false;
	std::vector<Point3d> pCloud;
	std::vector<Vec3b> pColours;
	cv::Mat colours;
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

	bool findNearbyPoint(cv::Point2f start, int width, int height, set<cv::Point2f>& set, cv::Point2f& hit);

	//object finder with naive edge finder
	bool processFrameData(cv::Mat data);

	bool update();

	void keyboardCallback(const cv::viz::KeyboardEvent& event, void* cookie);

};

class TCA_Audio
{

};

class TCA_Text
{

};