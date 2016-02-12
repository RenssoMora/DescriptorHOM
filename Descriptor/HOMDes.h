#ifndef HOMDES_H
#define HOMDES_H
#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"

struct	cuboid_dim{ int xi, yi, xf, yf;};
typedef cuboid_dim cutil_grig_point;
//w = frame width 
//h = frame height  
//cw = cuboid width
//ch = cuboid height
//ov_w = overlap_width
//ov_h = overlap_height
std::vector<cutil_grig_point> grid_generator( int w, int h, int cw, int ch, 
                                              int ov_w, int ov_h ){
	int wfin = w - cw + 1,
		  hfin = h - ch + 1;
	std::vector<cutil_grig_point> res;
	for (int i = 0; i < wfin; i += ov_w){
		for (int j = 0; j < hfin; j += ov_h){
			cutil_grig_point cuboid{ i, j, (i + cw - 1), (j + ch - 1) };
			res.push_back(cuboid);
		}
	}
	return res;
}
//
std::ostream & operator << (std::ostream & os, cuboid_dim c){
	os << "p1 " << c.xi << " " << c.yi << std::endl;
	os << "p2 " << c.xf << " " << c.yf;
	return os;
}
////===========================================================================
//main definitions 
typedef std::pair< cv::Mat_<float>, cv::Mat_<float> >	DesparMat;
typedef std::vector<DesparMat>                        DesvecParMat;
typedef std::vector<cutil_grig_point>                 CuboTypeCont;
typedef cv::Mat_<float>                               HistoType;
typedef std::pair<DesvecParMat, CuboTypeCont>	        DesInData;	//input data 
typedef std::vector<HistoType>                        DesOutData;	//output data

//=============================================================================
//descriptor magnitude orientation  
struct OFBasedDescriptorMO
{
	int		_orientNumBin,
        _magnitudeBin;
	float	_maxMagnitude,
			  _thrMagnitude;
	//____________________________________________________________________________
	
	void Describe(DesInData &in, DesOutData &out)
	{
		double	binRange		= 360 / _orientNumBin,
				binVelozRange	= _maxMagnitude / (float)_magnitudeBin;
		int		cubPos			= 0;
		for (auto & cuboid : in.second ){									//for each cuboid
			HistoType histogram(1, _orientNumBin * (_magnitudeBin + 1));
			histogram = histogram * 0;
			for (auto & imgPair : in.first){								// for each image
				for (int i = cuboid.xi; i <= cuboid.xf; ++i){				//for each row 
					for (int j = cuboid.yi; j <= cuboid.yf; ++j){			//for each pixel
						if (imgPair.second(i, j) > _thrMagnitude){			//large enough
							int p = (int)(imgPair.first(i, j) / binRange);
							int s = (int)(imgPair.second(i, j) / binVelozRange);
							if (s >= _magnitudeBin) s = _magnitudeBin;
							++histogram(0, p*(_magnitudeBin+1) + s);
						}
					}
				}
			}
			out[cubPos++].push_back(histogram);		
		}
	}

	OFBasedDescriptorMO(std::string file){
		cv::FileStorage fs(file, cv::FileStorage::READ);
		fs["descriptor_orientNumBin"] >> _orientNumBin;
		fs["descriptor_magnitudeBin"] >> _magnitudeBin;
		fs["descriptor_maxMagnitude"] >> _maxMagnitude;
		fs["descriptor_thrMagnitude"] >> _thrMagnitude;
	}
	OFBasedDescriptorMO(int orientB, int magnitudeB, float maxM, float thrM):
		_orientNumBin(orientB), 
		_magnitudeBin(magnitudeB),
		_maxMagnitude(maxM),
		_thrMagnitude(thrM)
	{}
};

///////////////////////////////////////////////////////////////////////////////
//=============================================================================
//main typedefs for opticalflow 
typedef std::vector< cv::Mat >							          OFdataType;
typedef std::pair< cv::Mat_<float>, cv::Mat_<float> >	OFparMat;
typedef std::vector<OFparMat>							            OFvecParMat;
//=============================================================================
//heritance for Optical flow 
struct OpticalFlowBase{	virtual void compute(OFdataType &, OFvecParMat &)= 0;};
//_____________________________________________________________________________

struct OpticalFlowOCV : public OpticalFlowBase{
	virtual void	compute(OFdataType & /*in*/, OFvecParMat & /*out*/);
};


static inline void FillPointsOriginal( std::vector<cv::Point2f> &vecPoints, 
									   cv::Mat  fr_A, cv::Mat  fr_a, int thr = 30)
{
	vecPoints.clear();
	cv::cvtColor(fr_A, fr_A, CV_BGR2GRAY);
	cv::cvtColor(fr_a, fr_a, CV_BGR2GRAY);
	//frame substraction
	cv::Mat fg = fr_A - fr_a;
	for (int i = 0; i < fg.rows; ++i)
	{
		for (int j = 0; j< fg.cols; ++j)
		{
			if (abs(fg.at<uchar>(i, j))  >thr)
				vecPoints.push_back(cv::Point2f((float)j, (float)i));
		}
	}
}

static inline void VecDesp2Mat( std::vector<cv::Point2f> &vecPoints, 
                                std::vector<cv::Point2f> &positions, 
								std::pair<cv::Mat_<float>, cv::Mat_<float> > & AMmat )
{
	float	magnitude,
			  angle,
			  catetoOpuesto,
			  catetoAdjacente;
	//...........................................................................
	for (int i = 0; i < (int)positions.size(); ++i)	{
		catetoOpuesto	= vecPoints[i].y - positions[i].y;
		catetoAdjacente = vecPoints[i].x - positions[i].x;
		//determining the magnite and the angle
		magnitude = sqrt((catetoAdjacente	* catetoAdjacente) +
						(catetoOpuesto		* catetoOpuesto));
		//signed or unsigned-------------------------------------------------------
		angle = (float)atan2f(catetoOpuesto, catetoAdjacente) * 180 / CV_PI;
		if (angle<0) angle += 360;

		AMmat.first((int)positions[i].y,  (int)positions[i].x) = angle;
		AMmat.second((int)positions[i].y, (int)positions[i].x) = magnitude;
	}
}

void OpticalFlowOCV::compute(OFdataType & in, OFvecParMat & out)
{
	assert(in.size()>1);
	std::vector<cv::Point2f> pointsprev, pointsnext, basePoints;
	std::vector<uchar>		status;
	cv::Mat					      err;
	cv::Size				      winSize(31, 31);
	cv::TermCriteria		  termcrit( cv::TermCriteria::COUNT | 
                                  cv::TermCriteria::EPS, 20, 0.3);
	int	rows,
			cols;
	//.......................................................
	rows = in[0].rows;
	cols = in[0].cols;
	for (size_t i = 0; i < in.size() - 1; ++i){
		FillPointsOriginal(pointsprev, in[i + 1], in[i]);
		cv::Mat angles(rows, cols, CV_32FC1, cvScalar(0.));
		cv::Mat magni(rows, cols, CV_32FC1, cvScalar(0.));
		OFparMat data;
		data.first = angles;
		data.second = magni;
		//computing optical flow por each pixel
		if (pointsprev.size()>0){
			cv::calcOpticalFlowPyrLK(in[i], in[i + 1], pointsprev, pointsnext,
				status, err, winSize, 3, termcrit, 0, 0.001);
			VecDesp2Mat(pointsnext, pointsprev, data);
		}
		/*cv::Mat sc = in[i+1].clone();
		for (auto &p : pointsprev){
			cv::circle(sc, p, 0.5, cv::Scalar(0, 255, 0));
		}
		for (auto &p : pointsnext){
			cv::circle(sc, cv::Point((int)p.x, (int)p.y), 0.5, cv::Scalar(255, 0, 255));
		}/**/
		out.push_back(data);
	}
	basePoints.clear();
}

//.............................................................................
//main standalone function describe
//This function receives a image set (cuboid- for instance 3 frames as temporal 
//length), and receives the n rectangles that you want to describe, it is im_..
//portant to note that in this example we are going to create a grid, but if...
//you have especifc points, you can create a rectangle patch centered in the...
//especific points. The function also receives a pointer tooptical flow base...
//class. Opencv optical flow and frame substraction is set by default.
//The function returns a vector with n histograms in outDes referenced variable 
void HOM( 
          DesOutData            &output,
          std::vector<cv::Mat>  &images, 
          std::vector<cutil_grig_point> &rects,
          int   orientbin,
          int   magnitudebin,
          float maxMagnitude,
          float thrMagnitude,
          OpticalFlowBase * of = new OpticalFlowOCV ){
  OFvecParMat outOf;
  of->compute(images, outOf);
  DesInData   input ( outOf, rects );
  output.clear();
  output.resize( rects.size() );
  OFBasedDescriptorMO Descriptor( orientbin, magnitudebin, maxMagnitude,
                                  thrMagnitude);
  Descriptor.Describe(input, output);
}

#endif