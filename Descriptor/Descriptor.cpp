// Descriptor.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"

#include "HOMDes.h"

int _tmain(int argc, char ** argv)
{
	int cuboid_width	= 30,
		  cuboid_height	= 30;

		
	cv::Mat	img1 = cv::imread("043.tif"),
			    img2 = cv::imread("045.tif"),
          img3 = cv::imread("047.tif");

  std::vector<cv::Mat> images{ img1, img2, img3 };

	std::vector<cutil_grig_point>	grid;
	grid = grid_generator(img1.rows, img1.cols,
		                    cuboid_width, cuboid_height,
		                    cuboid_width, cuboid_height );
	  
  DesOutData out;
  HOM(out, images, grid, 4, 6, 15, 0.1);
			
	return 0;
}

