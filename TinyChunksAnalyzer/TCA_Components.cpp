#include "stdafx.h"
#include <map>
#include <set>
#include <list>
#include "TCA_Components.h"
#include <opencv2/opencv.hpp>
using namespace std;

TCA_Video::TCA_Video()
{
	ready = true;
	if (!cap.open(701))
		ready = false;
	objects = 0;
	objectMap.empty();
	bool useNaiveEdge = false;
	bool useGraphSLAM = true;
	if (useGraphSLAM)
	{
		Mat frame;
		cap >> frame;
		gs.Initialize_LS_Graph_SLAM(frame);
		viewer = cv::viz::Viz3d("Point Cloud");
	}
	cv::namedWindow("VideoFeed1");
	cv::namedWindow("VideoFeed2");
	cv::namedWindow("VideoFeed3");
}

TCA_Video::~TCA_Video()
{

}

bool TCA_Video::findNearbyPoint(cv::Point2f start, int width, int height, set<cv::Point2f>& set, cv::Point2f& hit)
{
	for (int x = -3; x < 4; x++)
	{
		for (int y = -3; y < 4; y++)
		{
			cv::Point2f npoint(start.x + x, start.y + y);
			if ((x == 0 && y == 0) || npoint.x > width || npoint.y > height || npoint.x < 0 || npoint.y < 0) continue;
			//if hit
			if (set.count(npoint))
			{
				hit = npoint;
				return true;
			}
		}
	}
	return false;
}

//object finder with naive edge finder
bool TCA_Video::processFrameData(cv::Mat data)
{

	//for each row(this can be parallelized later)
	int width = data.cols;
	int height = data.rows;
	map<cv::Point2f, cv::Vec3b> changeMap;
	set<cv::Point2f> changeList;
	for (int rowIndex = 0; rowIndex < width; rowIndex++)
	{
		cv::Vec3b lastPixel = data.at<cv::Vec3b>(cv::Point(0, 0));
		//for each pixel
		for (int i = 0; i < height; i++)
		{
			cv::Vec3b pixel = data.at<cv::Vec3b>(cv::Point(rowIndex, i));
			//find a change point, mark it in the change array. change array is per each line of pixels and contains the colour change degree on each side(?)
			int matches = 0;
			for (int c = 0; c < 3; c++)
			{
				if (lastPixel.val[c] * colourFudgeMax > pixel.val[c] && lastPixel.val[c] * colourFudgeMin < pixel.val[c]) matches++;
			}

			if (matches < 3)
			{
				cv::Point2f tpoint = cv::Point(rowIndex, i);
				changeList.emplace(tpoint);
				changeMap.emplace(tpoint, pixel);
			}
			lastPixel = pixel;
		}

	}
	pSet = changeList;
	//while points are in list
	while (changeList.size() > 0)
	{
		//get top point
		cv::Point2f tpoint = *(changeList.begin());
		//set origin as top point
		cv::Point2f origin = tpoint;
		//create new object and add first point
		TCA_Object obj;
		obj.numpoints++;
		obj.points.push_back(tpoint);
		changeList.erase(tpoint);
		//set bounding box limits to origin
		obj.x = origin.x;
		obj.y = origin.y;
		obj.height = 1;
		obj.width = 1;
		//store if we've used the origin seek trick
		bool resetOrigin = false;
		//while still finding point
		bool running = true;
		while (running)
		{
			//check all immediate surrounding locations in change array
			cv::Point2f hit;
			bool foundPoint = findNearbyPoint(tpoint, width, height, changeList, hit);
			//if no hit go to origin and continue searching from there, set origin to top and top to found point
			if (!foundPoint && !resetOrigin)
			{
				resetOrigin = true;
				foundPoint = findNearbyPoint(origin, width, height, changeList, hit);
				origin = tpoint;
			}
			//if still no hit search whole array list for closest node with colour cv::Match (later junk)
			//if hit
			if (foundPoint)
			{
				//updated colour change average
				//add hit to object
				obj.numpoints++;
				obj.points.push_back(hit);
				//add line between top point and hit to object
				Line line;
				line.start = tpoint;
				line.end = hit;
				//colour pixel need to be determined by the relative spacing of the two points
				//line.colourDif = norm(data.at<Vec3b>(Point(hit.x + xp, hit.y + yp)), data.at<Vec3b>(Point(hit.x - xp, hit.y - yp)), CV_L2);
				obj.lines.push_back(line);
				//remove hit from list
				changeList.erase(hit);
				//changeMap.erase(hit); //no need for this, waste of time
				//expand bounding box
				if (hit.x > obj.width + obj.x) obj.width = hit.x - obj.x;
				if (hit.y > obj.height + obj.y) obj.height = hit.y - obj.y;
				if (hit.x < obj.x) obj.x = hit.x;
				if (hit.y < obj.y) obj.y = hit.y;
				//set top point to hit
				tpoint = hit;
			}
			//else
			else
			{
				running = false;
				//draw line between origin and top point
				Line line;
				line.start = tpoint;
				line.end = origin;
				//colour pixel need to be determined by the relative spacing of the two points
				//line.colourDif = norm(data.at<Vec3b>(Point(hit.x + xp, hit.y + yp)), data.at<Vec3b>(Point(hit.x - xp, hit.y - yp)), CV_L2);
				obj.lines.push_back(line);
				//set top point to empty
			}

		}
		objectMap.emplace(objects, obj);
		objects++;
	}
	return true;
}



bool TCA_Video::update()
{
	if (ready)
	{

		cv::Mat frame;
		cap >> frame;
		imshow("VideoFeed1", frame);
		if (frame.empty())
		{
			ready = false;
			return false;
		}

		if (useNaiveEdge)
		{
			cv::Mat edge;
			cv::Mat waste;
			cv::cvtColor(frame, edge, CV_BGR2GRAY);
			double otsu_thresh_val = cv::threshold(edge, waste, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
			cv::Canny(edge, edge, otsu_thresh_val * 0.2, otsu_thresh_val);
			imshow("VideoFeed2", edge);
			//get frame difference
			processFrameData(frame); //later processFrameData(changedFrame);
									 //draw each object's bounding box
			for (int i = 0; i < objects; i++)
			{
				cv::Scalar colour(0, 255, 0);
				//cv::Rect boundingRect(objectMap[i].x, objectMap[i].y, objectMap[i].width, objectMap[i].height);
				//rectangle(frame, boundingRect, colour);
				/*for(int j =0; j < objectMap[i].points.size(); j++)
					cv::circle(frame, cv::Point(objectMap[i].points[j].x, objectMap[i].points[j].y), 2, colour);*/
				for (auto f : pSet) {
					cv::circle(frame, f, 2, colour);
				}
			}
			imshow("VideoFeed3", frame);
			cv::waitKey(10000);
		}
		else if (useGraphSLAM)
		{
			//runs the graph slam process for the given frame, and returns the current position of the camera as an SE3 object
			GraphSLAMer::SE3 location = gs.LS_Graph_SLAM(frame);

			//get back current list of 3d points
			pCloud = gs.get3dPoints();
			
			if (!widgetSet)
			{
				widgetSet = true;
				viz::WCloud cloud_widget = viz::WCloud(pCloud, viz::Color::green());
				viewer.showWidget("Point Cloud", cloud_widget);
				viewer.spinOnce();
			}
			
		}
		return true;
	}
	else
	{
		return false;
	}
}