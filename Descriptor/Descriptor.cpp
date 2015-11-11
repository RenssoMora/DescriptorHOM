// Descriptor.cpp : Defines the entry point for the console application.
//


#include "HOMDes.h"

int _tmain(int argc, char ** argv)
{
	int cuboid_width	= 30,
		cuboid_height	= 30;
	//Optical flow class extractor, you may choose other
	OpticalFlowOCV ofOCV; 
	cv::Mat	img1 = cv::imread("043.tif"),
			img2 = cv::imread("045.tif"),
			img3 = cv::imread("047.tif");
	OFdataType	in{ img1, img2, img3 };
	OFvecParMat	outOf;
	ofOCV.compute(in, outOf);
	//__________________________________________________
	DesInData	input;
	std::vector<cutil_grig_point>	grid;
	grid = grid_generator(img1.rows, img1.cols,
		   cuboid_width, cuboid_height,
		   cuboid_width, cuboid_height );
	input.first		= outOf;
	input.second	= grid;
	DesOutData  output(grid.size());
	//__________________________________________________
	OFBasedDescriptorMO Descriptor( /*orientbin*/4, 
									/*magnitudebin*/6, 
									/*MaxMagnitude*/15, 
									/*Magnitude thr*/0.1);
	Descriptor.Describe(input, output);
			
	return 0;
}

