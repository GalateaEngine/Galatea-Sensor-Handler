// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once
#include "targetver.h"

#include <stdio.h>
#include <tchar.h>
#include <opencv2\opencv.hpp>

namespace std {
	inline bool operator<(const cv::Point a, const cv::Point b)
	{
		if (a.x != b.x) return a.x < b.x;
		else return a.y < b.y;
	}

	inline bool operator==(const cv::Point a, const cv::Point b)
	{
		return (a.x == b.x) && (a.y == b.y);
	}
}

// TODO: reference additional headers your program requires here
