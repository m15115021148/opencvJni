
#include <jni.h>
#include "cubic_inc.h"
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <iostream>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>

//#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::ocl;
using namespace xfeatures2d;

Mat empty;
bool initialized_already = false;
// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.0f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "cylindrical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.40f;
string seam_find_type = "voronoi";
//string seam_find_type = "gc_colorgrad";
int blend_type = Blender::FEATHER;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;
struct BlendNPara
{
	Mat dst_, dst_mask_;
	Rect dst_roi_;
	//表示最终得到的全景图像的矩形变量
	// float sharpness_;
	// UMat weight_map_;
	// UMat dst_weight_map_;
};
BlendNPara Blend_No_struct;

void blend_prepare(const std::vector<Point> &corners,
	const std::vector<Size> &sizes,
	BlendNPara *Blend_No_struct)
{
	Rect dst_roi = resultRoi(corners, sizes);
	Blend_No_struct->dst_.create(dst_roi.size(), CV_16SC3);
	Blend_No_struct->dst_.setTo(Scalar::all(0));
	Blend_No_struct->dst_mask_.create(dst_roi.size(), CV_8U);
	Blend_No_struct->dst_mask_.setTo(Scalar::all(0));
	Blend_No_struct->dst_roi_ = dst_roi;
}

void blend_feed(Mat _img,
	Mat _mask,
	Point tl,
	BlendNPara *Blend_No_struct)
{
	//Mat img      = _img.getMat();
	//Mat mask     = _mask.getMat();
	//Mat dst      = Blend_No_struct->dst_.getMat(ACCESS_RW);
	//Mat dst_mask = Blend_No_struct->dst_mask_.getMat(ACCESS_RW);

	//CV_Assert(img.type() == CV_16SC3);
	//CV_Assert(mask.type() == CV_8U);
	int dx = tl.x - Blend_No_struct->dst_roi_.x;
	int dy = tl.y - Blend_No_struct->dst_roi_.y;

	for (int y = 0; y < _img.rows; ++y)
	{
		const Point3_<short> *src_row = _img.ptr<Point3_<short> >(y);
		Point3_<short> *dst_row = Blend_No_struct->dst_.ptr<Point3_<short> >(dy + y);
		const uchar *mask_row = _mask.ptr<uchar>(y);
		uchar *dst_mask_row = Blend_No_struct->dst_mask_.ptr<uchar>(dy + y);

		for (int x = 0; x < _img.cols; ++x)
		{
			if (mask_row[x])
				dst_row[dx + x] = src_row[x];
			dst_mask_row[dx + x] |= mask_row[x];
		}
	}
}
void blend_blend(InputOutputArray dst,
	InputOutputArray dst_mask,
	BlendNPara *Blend_No_struct
)
{
	// UMat mask;
	// compare(Blend_No_struct->dst_mask_, 0, mask, CMP_EQ);
	// Blend_No_struct->dst_.setTo(Scalar::all(0), mask);
	// dst.assign(Blend_No_struct->dst_);
	// dst_mask.assign(Blend_No_struct->dst_mask_);
	// Blend_No_struct->dst_.release();
	// Blend_No_struct->dst_mask_.release();
	// void Blender::blend(Mat &dst, Mat &dst_mask)

	//dst和dst_mask表示最终得到的全景图像和掩码

	//UMat mask;

	//Blend_No_struct->dst_mask_.copyTo(mask);    //赋值
	//compare(Blend_No_struct->dst_mask_, 0, mask, CMP_EQ);
	//Blend_No_struct->dst_.setTo(Scalar::all(0), mask);    //为掩码部分赋0值
	Blend_No_struct->dst_.copyTo(dst);    //赋值
	//Blend_No_struct->dst_mask_.copyTo(dst_mask);    //赋值
	Blend_No_struct->dst_.release();    //释放内存
	//Blend_No_struct->dst_mask_.release();    //释放内容

}

vector<Point> corners(2);
vector<UMat> masks_warped(2);
vector<UMat> images_warped(2);
vector<Size> sizes(2);
vector<UMat> masks(2);

Mat img_warped, img_warped_s;
Mat dilated_mask, seam_mask, mask, mask_warped;
vector<Mat> mask_final(2);
vector<UMat> xmap(2), ymap(2);
Ptr<Blender> blender;
Ptr<Timelapser> timelapser;
Rect dst_roi;
string imgresult, imgleft, imgright;
//double compose_seam_aspect = 1;
double compose_work_aspect = 1;


#ifdef CUBIC_LOG_TAG
#undef CUBIC_LOG_TAG
#endif //CUBIC_LOG_TAG
#define CUBIC_LOG_TAG  "OpenCV"


int checkSupportOpenCL();

Mat splice_image(const string &img1, const string &img2, Mat frame1, Mat frame2);

jdouble mergeImage(JNIEnv *env, jobject type,
							  jstring img1_, jstring img2_,
							  jstring path_) {
	const char *img1 = env->GetStringUTFChars(img1_, 0);
	const char *img2 = env->GetStringUTFChars(img2_, 0);												  
	const char *path = env->GetStringUTFChars(path_, 0);		
	
	checkSupportOpenCL();
	
	setUseOpenCL(true);
	
	double time = getTickCount();	
	double t_s = getTickCount();
	
	vector<UMat>img_names;

	UMat image01 = imread(img1).getUMat(ACCESS_RW);
	cvtColor(image01, image01, CV_BGRA2BGR);

	UMat image02 = imread(img2).getUMat(ACCESS_RW);
	cvtColor(image02, image02, CV_BGRA2BGR);

	img_names.push_back(image01);
	img_names.push_back(image02);
	
	Stitcher stitcher = Stitcher::createDefault(true);

	// 默认是0.6,最大值1最慢，此方法用于特征点检测阶段，如果找不到特征点，要调高
	stitcher.setRegistrationResol(0.6);
 
	//stitcher.setSeamEstimationResol(0.1); // 默认是0.1
	//stitcher.setCompositingResol(-1);     // 默认是-1，用于特征点检测阶段，找不到特征点的话，改-1
	stitcher.setPanoConfidenceThresh(1);  // 默认是1,见过有设0.6和0.4的
	stitcher.setWaveCorrection(true);    // 默认是true，为加速选false，表示跳过WaveCorrection步骤
	//还可以选detail::WAVE_CORRECT_VERT ,波段修正(wave correction)功能（水平方向/垂直方向修正）。因为setWaveCorrection设的false，此语句没用
	stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
 
	// 找特征点surf算法，此算法计算量大,但对刚体运动、缩放、环境影响等情况下较为稳定
	stitcher.setFeaturesFinder(new detail::SurfFeaturesFinder);
	//stitcher.setFeaturesFinder(new detail::OrbFeaturesFinder);// ORB
 
	// Features matcher which finds two best matches for each feature and leaves the best one only if the ratio between descriptor distances is greater than the threshold match_conf.
	// match_conf默认是0.65，选太大了没特征点
	detail::BestOf2NearestMatcher* matcher = new detail::BestOf2NearestMatcher(false, match_conf);
	stitcher.setFeaturesMatcher(matcher);
 
	// Rotation Estimation,It takes features of all images, pairwise matches between all images and estimates rotations of all cameras.
	//Implementation of the camera parameters refinement algorithm which minimizes sum of the distances between the rays passing through the camera center and a feature,
	//这个耗时短
	stitcher.setBundleAdjuster(new detail::BundleAdjusterRay);
	//Implementation of the camera parameters refinement algorithm which minimizes sum of the reprojection error squares.
	//stitcher.setBundleAdjuster(new detail::BundleAdjusterReproj);
 
	//Seam Estimation
	//Minimum graph cut-based seam estimator
	//stitcher.setSeamFinder(new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR));//默认就是这个
	//stitcher.setSeamFinder(new detail::GraphCutSeamFinder(detail::GraphCutSeamFinderBase::COST_COLOR_GRAD));//GraphCutSeamFinder的第二种形式
	//啥SeamFinder也不用，Stub seam estimator which does nothing.
	stitcher.setSeamFinder(new detail::NoSeamFinder);
	//Voronoi diagram-based seam estimator.
	//stitcher.setSeamFinder(new detail::VoronoiSeamFinder);
 
	//exposure compensators曝光补偿
	//stitcher.setExposureCompensator(new detail::BlocksGainCompensator);//默认的就是这个
	//不要曝光补偿
	stitcher.setExposureCompensator(new detail::NoExposureCompensator);
	//Exposure compensator which tries to remove exposure related artifacts by adjusting image intensities
	//stitcher.setExposureCompensator(new detail::detail::GainCompensator);
	//Exposure compensator which tries to remove exposure related artifacts by adjusting image block intensities 
	//stitcher.setExposureCompensator(new detail::detail::BlocksGainCompensator); 
 
	// 边缘Blending
	//stitcher.setBlender( new detail::MultiBandBlender(false) );// 默认使用这个,use gpu
	//Simple blender which mixes images at its borders
	stitcher.setBlender(new detail::FeatherBlender);// 这个简单，耗时少
 
	// 拼接方式，柱面？球面OR平面？默认为球面
	//stitcher.setWarper(new PlaneWarper);
	//stitcher.setWarper(new SphericalWarper);
	//stitcher.setWarper(new CylindricalWarper);
 
	// 开始计算变换
	Stitcher::Status status = stitcher.estimateTransform(img_names);
	if (status != Stitcher::OK)
	{
		std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
		LOGE("Can't stitch images, error code = %d",int(status));
		return -1;
	}
	else
	{
		std::cout << "Estimate transform complete" << std::endl;
		LOGE("Estimate transform complete");
	}
	
	
	
	int id = 1;
	
	while (true) {
		UMat pano;
		img_names.clear();
		
		img_names.push_back(image01);
		img_names.push_back(image02);
		
		t_s = getTickCount();
		status = stitcher.composePanorama(img_names,pano);
		LOGD("match t_s splice time=%f\n", (getTickCount() - t_s)/getTickFrequency());
		
		char name[521] = {0};
		sprintf(name, "%s/image/%0d.jpg", path, id);
		
		imwrite(name, pano);
		
		id++;
	}
 
	LOGD("match all time=%f\n", (getTickCount() - time)/getTickFrequency());
	
	env->ReleaseStringUTFChars(img1_, img1);
	env->ReleaseStringUTFChars(img2_, img2);
	env->ReleaseStringUTFChars(path_, path);

	return time;
}

jdouble playVideo(JNIEnv *env, jobject type,
							  jstring img1_, jstring img2_,
							  jstring firstPath_, jstring secondPath_,
							  jstring path_) {
	const char *img1 = env->GetStringUTFChars(img1_, 0);
	const char *img2 = env->GetStringUTFChars(img2_, 0);												  
	const char *firstPath = env->GetStringUTFChars(firstPath_, 0);
	const char *secondPath = env->GetStringUTFChars(secondPath_, 0);
	const char *path = env->GetStringUTFChars(path_, 0);
	
	LOGD("img1 path : %s",img1);
	LOGD("img2 path : %s",img2);
	
	int ret = checkSupportOpenCL();

	if (ret == 0){
		cv::ocl::setUseOpenCL(true);
	}
	
	
	double time = getTickCount();
	double splice_time = getTickCount();
	double same_time = getTickCount();
	
	vector<String> fn_left;
	vector<String> fn_right;

	glob(firstPath, fn_left, false);
	glob(secondPath, fn_right, false);
	
	size_t count = fn_left.size();
	
	size_t index = fn_left.size();

	LOGD("read img size = %d",count);
	
	Mat merge_frame;
	
	size_t n = 1;

	for (size_t i = 1; i <= count; i++)
	{
		Mat frame1,frame2;
		stringstream str;
		str << i << ".jpg";
		string left = firstPath + str.str();
		string right = secondPath + str.str();
		
		frame1 = imread(left);
		frame2 = imread(right);

		if (frame1.empty() || frame2.empty() ){
			LOGE("read mat frame is empty");
			break;
		}
		
		
		splice_time = getTickCount();
		
		if (i == index){
			same_time = getTickCount();
		}
		

		merge_frame = splice_image(img1,img2,frame1,frame2);
	

		if (merge_frame.empty() ){
		   LOGE("splice image is empty...");
		   return -1;
		}

		splice_time = getTickCount() - splice_time;
		splice_time /= getTickFrequency();
		LOGD("splice image splice_time=%f\n", splice_time);

		char name[512] = {0};
		sprintf(name, "%s/merge/%0d.jpg", path, n);

		imwrite(name, merge_frame);
		
		n++;

	}

	time = getTickCount() - time;
	time /= getTickFrequency();
	LOGD("match end time=%f\n", time);
	
	LOGD("before 100 all time=%f\n",same_time);
	
	env->ReleaseStringUTFChars(img1_, img1);
	env->ReleaseStringUTFChars(img2_, img2);
	env->ReleaseStringUTFChars(firstPath_, firstPath);
	env->ReleaseStringUTFChars(secondPath_, secondPath);
	env->ReleaseStringUTFChars(path_, path);

	return time;
};

int checkSupportOpenCL(){
	//opencl is support?
	try {
		if (!cv::ocl::haveOpenCL()){
			LOGD("OpenCL is not availble");
		} else {
			LOGD("OpenCL is avaible");
		}
		if (cv::ocl::useOpenCL()){
			LOGD("use OpenCL");
		} else {
			LOGD("don't use OpenCL");
		}
		cv::ocl::Context context;
		if (!context.create(cv::ocl::Device::TYPE_GPU)){
			LOGD("Failed creating the context...");
			return -1;
		} else {
			LOGD("ocl::Context is OK");
		}

		LOGD(" %u GPU devices are detected.",context.ndevices());
		for (int i = 0; i < context.ndevices(); i++){
			cv::ocl::Device device = context.device(i);
			LOGD("name: %s",device.name().c_str());
			if (device.available()){
				LOGD("device is avaible");
			} else {
				LOGD("devive is not avaible");
			}
			if (device.imageSupport()){
				LOGD("device support image");
			} else {
				LOGD("device doesn't support image");
			}
			LOGD("OpenCL_C_Version     : %s" ,device.OpenCL_C_Version().c_str());
		}
	} catch (cv::Error::Code& e) {
		LOGE("cv::Error::Code %d", e);
		return -1;
	}
	return 0;
};

Mat splice_image(const string &img1, const string &img2, Mat frame1, Mat frame2)
{
	if(!initialized_already)
{
	BlendNPara FeatherFeature;

#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

	bool stop(false);
	//Mat frame1;
	//Mat frame2;
	//Mat frameCalibration;
	//Size imageSize;
	//string imgname;
	//string calimgname;
	//string calimgname1;
	//Mat view, rview, map1left, map2left;
	//Mat map1right, map2right;
#ifdef HAVE_OPENCL
	cout << "HAVE_OPENCL" << endl;
#endif


#if 0
	cv::setBreakOnError(true);
#endif
	initialized_already = true;
	// int retval = parseCmdArgs(argc, argv);
	// if (retval)
	  //   return retval;
	
	img_names.push_back(img1);
	img_names.push_back(img2);
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return empty;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	LOGLN("Finding features...");
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	Ptr<FeaturesFinder> finder;
	if (features_type == "surf")
	{
#ifdef HAVE_OPENCV_XFEATURES2D
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			finder = makePtr<SurfFeaturesFinderGpu>();
		else
#endif
			finder = makePtr<SurfFeaturesFinder>();
	}
	else if (features_type == "orb")
	{
		finder = makePtr<OrbFeaturesFinder>();
	}
	else if (features_type == "sift") {
		finder = makePtr<SiftFeaturesFinder>();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return empty;
	}

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return empty;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

		resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	//LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	LOG("Pairwise matching");
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	else if (range_width == -1)
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	else
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();

	//LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<String> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return empty;
	}

	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();

	vector<CameraParams> cameras;
	if (!(*estimator)(features, pairwise_matches, cameras))
	{
		cout << "Homography estimation failed.\n";
		return empty;
	}

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray") adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine") adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return empty;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras))
	{
		cout << "Camera parameters adjusting failed.\n";
		return empty;
	}

	// Find median focal length

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());

	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarperGpu>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarperGpu>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarperGpu>();
	}
	else
#endif
	{
		if (warp_type == "plane")
			warper_creator = makePtr<cv::PlaneWarper>();
		else if (warp_type == "affine")
			warper_creator = makePtr<cv::AffineWarper>();
		else if (warp_type == "cylindrical")
			warper_creator = makePtr<cv::CylindricalWarper>();
		else if (warp_type == "spherical")
			warper_creator = makePtr<cv::SphericalWarper>();
		else if (warp_type == "fisheye")
			warper_creator = makePtr<cv::FisheyeWarper>();
		else if (warp_type == "stereographic")
			warper_creator = makePtr<cv::StereographicWarper>();
		else if (warp_type == "compressedPlaneA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlaneA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA2B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "compressedPlanePortraitA1.5B1")
			warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniA2B1")
			warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniA1.5B1")
			warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
		else if (warp_type == "paniniPortraitA2B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
		else if (warp_type == "paniniPortraitA1.5B1")
			warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
		else if (warp_type == "mercator")
			warper_creator = makePtr<cv::MercatorWarper>();
		else if (warp_type == "transverseMercator")
			warper_creator = makePtr<cv::TransverseMercatorWarper>();
	}

	if (!warper_creator)
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return empty;
	}
	float scale_warper_input = static_cast<float>(warped_image_scale * seam_work_aspect);
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(scale_warper_input));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<UMat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	//LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = makePtr<detail::NoSeamFinder>();
	else if (seam_find_type == "voronoi")
		seam_finder = makePtr<detail::VoronoiSeamFinder>();
	else if (seam_find_type == "gc_color")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#ifdef HAVE_OPENCV_CUDALEGACY
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = makePtr<detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
	if (!seam_finder)
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return empty;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);
	/*
	vector<Mat> dilate_img(2);
	vector<Mat> images_warped_2(2);
	for (int i = 0; i < 2; i++)
	{
		images_warped[i].copyTo(images_warped_2[i]);
	}
	//通过canny边缘检测，得到掩码边界，其中有一条边界就是接缝线
	for (int k = 0; k < 2; k++)
		Canny(images_warped[k], images_warped_2[k], 3, 9, 3);

	//为了使接缝线看得更清楚，这里使用了膨胀运算来加粗边界线
	Mat element = getStructuringElement(MORPH_RECT, Size(10, 10));    //定义结构元素

	for (int k = 0; k < 2; k++)    //遍历两幅图像
	{
		dilate(images_warped_2[k], dilate_img[k], element);    //膨胀运算
		//在映射变换图上画出接缝线，在这里只是为了呈现出的一种效果，所以并没有区分接缝线和其他掩码边界
		for (int y = 0; y < images_warped_2[k].rows; y++)
		{
			for (int x = 0; x < images_warped_2[k].cols; x++)
			{
				if (dilate_img[k].at<uchar>(y, x) == 255)    //掩码边界
				{
					//((Mat)images_warped[k]).at<Vec3b>(y, x)[0] = 255;
					images_warped_2[k].at<Vec3b>(y, x)[1] = 0;
					images_warped_2[k].at<Vec3b>(y, x)[2] = 255;
				}
			}
		}
	}

	imwrite("seam1.jpg", images_warped_2[0]);    //存储图像
	imwrite("seam2.jpg", images_warped_2[1]);
	*/
	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	LOGLN("Compositing...");
#if ENABLE_LOG
#endif
	bool is_registration_finished = false;
	Mat Proto_frame1, Proto_frame2;
	//	string imgname, imgleft, imgright;

	Proto_frame1 = imread(img1);
	Proto_frame2 = imread(img2);

	//t = getTickCount();
	

	Mat K;

	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		LOGLN("Compositing image #" << indices[img_idx] + 1);

		// Read image and resize it if necessary
		full_img = imread(img_names[img_idx]);
		if (!is_compose_scale_set)
		{
			if (compose_megapix > 0)
				compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
			is_compose_scale_set = true;

			// Compute relative scales
			//compose_seam_aspect = compose_scale / seam_scale;
			compose_work_aspect = compose_scale / work_scale;

			// Update warped image scale
			warped_image_scale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_image_scale);

			// Update corners and sizes
			for (int i = 0; i < num_images; ++i)
			{
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1)
				{
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				//Mat K;
				cameras[i].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cameras[i].R);
				corners[i] = roi.tl();
				sizes[i] = roi.size();
			}
		}
		if (abs(compose_scale - 1) > 1e-1)
			resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
		else
			img = full_img;
		full_img.release();
		Size img_size = img.size();

		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		//warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		//UMat uxmap, uymap;
		//Rect dst_roi = warper->buildMaps(img.size(), K, cameras[img_idx].R, uxmap, uymap);
		//img.create(dst_roi.height + 1, dst_roi.width + 1, img.type());
		//remap(img, img_warped, uxmap, uymap, INTER_NEAREST, BORDER_CONSTANT);

		UMat Uimg;
		img.copyTo(Uimg);
		//namedWindow("Uimg",WINDOW_NORMAL );
		//imshow("Uimg", Uimg);
		//waitKey();
		/*
		Mat Kt = (Mat_<float>(3, 3) <<
			4610.0337, 0, 2136.7131,
			0, 4610.0337, 1422.9744,
			0, 0, 1);
		vector<UMat> Rt(2);


		Mat Rt_temp1 = (Mat_<float>(3, 3) << 0.98261827, -0.023412688, -0.18415552,
			6.0605774e-09, 0.992015, -0.12612024,
			0.18563788, 0.12392807, 0.97477204
			);
		Mat Rt_temp2 = (Mat_<float>(3, 3) << 0.98255295, 0.019991236, 0.18490513,
			-7.4505806e-09, 0.99420613, -0.10748972,
			-0.18598272, 0.10561435, 0.97686023);
		Rt_temp1.copyTo(Rt[0]);
		Rt_temp2.copyTo(Rt[1]);
		*/
		//cout << "KR start:" << endl;
		//cout << K << endl;
		//cout << cameras[img_idx].R << endl;
		//Rect dst_roi = warper->buildMaps(Uimg.size(), Kt, Rt[img_idx], xmap, ymap);
		dst_roi = warper->buildMaps(Uimg.size(), K, cameras[img_idx].R, xmap[img_idx], ymap[img_idx]);
		img_warped.create(dst_roi.height + 1, dst_roi.width + 1, Uimg.type());
		float t = getTickCount();
		remap(Uimg, img_warped, xmap[img_idx], ymap[img_idx], INTER_LINEAR, BORDER_REFLECT);
		//LOGLN("remap time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		//namedWindow("remap", WINDOW_AUTOSIZE);
		//imshow("remap", img_warped);
		//namedWindow("remapUimg", WINDOW_NORMAL);
		//imshow("remapUimg", img_warped);
		//waitKey();
		//dst_roi.tl();

		//namedWindow("img_warped", WINDOW_NORMAL);
		//imshow("img_warped", img_warped);
		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		//warper->warp(mask, Kt, Rt[img_idx], INTER_NEAREST, BORDER_CONSTANT, mask_warped);
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);
		//buildMaps
		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();
		//namedWindow("img_warped_s", WINDOW_NORMAL);
		//imshow("img_warped_s", img_warped_s);
		//remap(srcImage, resultImage, xMapImage, yMapImage, CV_INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
		/*
		这部分代码的关键在于找到warper->warp函数中的remap部分原理
		然后重写这个函数为一片代码段
		同时仅讲remap函数作为循环部分
		其余部分全部并入初始化
		 */
		dilate(masks_warped[img_idx], dilated_mask, Mat());
		//imshow("masks_warped1", masks_warped[1]);
		//imshow("dilated_mask", dilated_mask);
		resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
		mask_warped = seam_mask & mask_warped;
		mask_final[img_idx] = mask_warped;
		if (!blender && !timelapse)
		{
			//第一次运行进入此条件分支
			blender = Blender::createDefault(blend_type, 1);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);
			else if (blend_type == Blender::MULTI_BAND)
			{

				//该分支只负责打印Log
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(2));
				//mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				//fb->setSharpness(1.f / blend_width);
				fb->setSharpness(1.f / blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			LOGLN(" blender, prepare: GET IN");
			blender->prepare(corners, sizes);
		}
		/*
			else if (!timelapser && timelapse)
			{
				timelapser = Timelapser::createDefault(timelapse_type);
				timelapser->initialize(corners, sizes);
			}*/

			// Blend the current image
		if (0)
		{
			timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
			String fixedFileName;
			size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
			if (pos_s == String::npos)
			{
				fixedFileName = "fixed_" + img_names[img_idx];
			}
			else
			{
				fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
			}
			imwrite(fixedFileName, timelapser->getDst());
		}
		else
		{
			LOGLN(" blender, feed: GET IN");
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}
	}

	if (!timelapse)
	{
		Mat result, result_mask;
		LOGLN(" blender, blend: GET IN");
		blender->blend(result, result_mask);
		//LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
		imwrite("keepfire.jpg", result);
	}
}
	//LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
	int length = 200;
	//img_warped_loop
	float Time_blend = 0;
	float Time_remap = 0;
	UMat Uimg;
	Mat result, result_mask;
	vector<Mat> full_imgt(2);
	vector<Mat> img_warped_loop(2);
	vector<Mat> img_input1(length);
	vector<Mat> img_input2(length);
	double feed_sumup = 0;
	
	//init end
	
	
	
		double t = getTickCount();
		//full_imgt[0] = img_input1[step_turn];
		//full_imgt[1] = img_input2[step_turn];
		full_imgt[0] = frame1;
		full_imgt[1] = frame2;
		//imgresult = to_string(step_turn) + ".jpg";

		blend_prepare(corners, sizes, &Blend_No_struct);
		//LOGLN("blend prepare time: " << ((getTickCount() - preparetime) / getTickFrequency()) << " sec");
		for (int i = 0; i < 2; i++)
		{
			//Mat K;
			//full_imgt[i].copyTo(Uimg);
			//full_imgt("Compositing image #" << i + 1);
			//cameras[i].K().convertTo(K, CV_32F);
			//dst_roi = warper->buildMaps(Uimg.size(), K, cameras[i].R, xmap[i], ymap[i]);
			//img_warped.create(dst_roi.height + 1, dst_roi.width + 1, Uimg.type());
			//double tremap = getTickCount();
			remap(full_imgt[i], img_warped, xmap[i], ymap[i], INTER_NEAREST, BORDER_CONSTANT);
			//remap(Uimg, img_warped, xmap[i], ymap[i], INTER_NEAREST, BORDER_CONSTANT);
			//Time_remap += (getTickCount() - tremap) / getTickFrequency();
			//LOGLN("remap turn time: " << ((getTickCount() - tremap) / getTickFrequency()) << " sec");
			if (i == 1) {
			//	LOGLN("remap: " << (Time_remap) << " sec");
				Time_remap = 0;

			}
			//namedWindow("full_img", WINDOW_AUTOSIZE);
			//warper->warp(full_img[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
			img_warped.convertTo(img_warped_loop[i], CV_16S);
			//namedWindow("img_warped_loop1", WINDOW_AUTOSIZE);
			//imwrite("undermask" + to_string(i) + ".jpg", img_warped_loop[i]);
			//imshow("img_warped_loop1", img_warped_loop[i]);
			//waitKey();
			double tblend = getTickCount();
			//if (i == 0)
			blend_feed(img_warped_loop[i], mask_final[i], corners[i], &Blend_No_struct);
			feed_sumup += getTickCount() - tblend;
			if (i == 1)
			{
				//float tblend_blend = getTickCount();
				blend_blend(result, result_mask, &Blend_No_struct);
				//imwrite(imgresult, result);
				//imshow("result", result);
				//LOGLN("blender: " << Time_blend << " sec");
				//Time_blend = 0;
				//LOGLN("blender blend: " << (getTickCount() - tblend_blend) / getTickFrequency() << " sec");
				LOGLN("blender feed: " << (getTickCount() - tblend) / getTickFrequency() << " sec");
				feed_sumup = 0;
			}
			//Time_blend += (getTickCount() - tblend) / getTickFrequency();
			//waitKey();
		}
		//LOGLN("blend all time: " << ((getTickCount() - preparetime) / getTickFrequency()) << " sec");
		//LOGLN("loop stitching time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl);

		return result;
	//waitKey();
	return empty;
};

//------------------------------------jni loaded----------------------------------------------------------
JNIEXPORT const char *classPathNameRx = "com/sensology/opencv/OpenUtil";

static JNINativeMethod methodsRx[] = { 
	{"playVideo","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)D",(void*)playVideo},
	{"mergeImage","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)D",(void*)mergeImage},
};

static jint registerNativeMethods(JNIEnv* env, const char* className,JNINativeMethod* gMethods, int numMethods){
    jclass clazz;

    clazz = env->FindClass(className);
    if (env->ExceptionCheck()) {
        env->ExceptionDescribe();
        env->ExceptionClear();
    }
    if (clazz == NULL) {
        LOGE("Native registration unable to find class '%s'", className);
        return JNI_FALSE;
    }
    if (env->RegisterNatives(clazz, gMethods, numMethods) < 0) {
        LOGE("RegisterNatives failed for '%s'", className);
        return JNI_FALSE;
    }

    LOGD("%s, success\n", __func__);
    return JNI_TRUE;
}

static jint registerNatives(JNIEnv* env){
    jint ret = JNI_FALSE;

    if (registerNativeMethods(env, classPathNameRx, methodsRx, sizeof(methodsRx) / sizeof(methodsRx[0]))) {
        ret = JNI_TRUE;
    }

    LOGD("%s, done\n", __func__);
    return ret;
}

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved){
	JNIEnv* env;
	LOGI("JNI_OnLoad");
	
    if (vm->GetEnv((void**)&env, JNI_VERSION_1_6) != JNI_OK) {
		LOGE("ERROR: GetEnv failed");
        return JNI_ERR; 
    }
	
	if (registerNatives(env) != JNI_TRUE) {
        LOGE("ERROR: registerNatives failed");
        return JNI_ERR;
    }
	
    return  JNI_VERSION_1_6;
	
}

