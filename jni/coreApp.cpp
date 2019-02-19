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

#define ENABLE_LOG 0
#define LOG(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;
using namespace cv::ocl;

vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.4f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
bool timelapse = false;
int range_width = -1;
Mat empty;
bool initialized_already = false;


//---init
int num_images = 2;
Mat full_img, img;
vector<ImageFeatures> features(num_images);
vector<Mat> images(num_images);
vector<Size> full_img_sizes(num_images);
double seam_work_aspect = 1;
vector<int> indices ;
vector<Mat> img_subset;
vector<String> img_names_subset;
vector<Size> full_img_sizes_subset;
bool is_registration_finished = false;
Mat Proto_frame1, Proto_frame2;
string imgname;
Ptr<RotationWarper> warper;
double work_scale = 1, seam_scale = 1, compose_scale = 1;
bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
float warped_image_scale;
Ptr<WarperCreator> warper_creator;
vector<CameraParams> cameras;

vector<Point> corners(num_images);
vector<UMat> masks_warped(num_images);
vector<UMat> images_warped(num_images);
vector<Size> sizes(num_images);
vector<UMat> masks(num_images);


#ifdef CUBIC_LOG_TAG
#undef CUBIC_LOG_TAG
#endif //CUBIC_LOG_TAG
#define CUBIC_LOG_TAG  "OpenCV"


int checkSupportOpenCL();

void cut_img(Mat src_img, int m, int n, vector<Mat> ceil_img,const char *path);

Mat splice_image(const string &img1, const string &img2, Mat frame1, Mat frame2);

jdouble playVideo(JNIEnv *env, jobject type,
							  jstring img1_, jstring img2_,
							  jstring videoPath_,
							  jstring path_) {
	const char *img1 = env->GetStringUTFChars(img1_, 0);
	const char *img2 = env->GetStringUTFChars(img2_, 0);												  
	const char *videoPath = env->GetStringUTFChars(videoPath_, 0);
	const char *path = env->GetStringUTFChars(path_, 0);
	
	LOGD("video path --> %s",videoPath);
	
	int ret = checkSupportOpenCL();

	if (ret == 0){
		cv::ocl::setUseOpenCL(true);
	}
	
	double time = getTickCount();

	VideoCapture capture(videoPath);

	if (!capture.isOpened()){
		LOGE("fail open video");
		return -1;
	}

	int id = 1;
	Mat merge_frame;
	vector<Mat> cut_frame;

	while (true) {
		LOGD("start --> read frame =%d",id);

		Mat frame;
		cut_frame.clear();

		bool f_success = capture.read(frame);

		if (!f_success){
			LOGE("read mat frame is empty");
			break;
		}
		
		cut_img(frame,1,2,cut_frame,path);
		
		merge_frame = splice_image(img1,img2,cut_frame[0],cut_frame[1]);

		if (merge_frame.empty() ){
		   LOGE("splice image is empty...");
		   return -1;
		}
		
		char name[512] = {0};
		sprintf(name, "%s/merge/%0d.jpg", path, id);

		imwrite(name, frame);

		id++;

	}
	capture.release();

	time = getTickCount() - time;
	time /= getTickFrequency();
	LOGD("match end time=%f\n", time);
		
	env->ReleaseStringUTFChars(img1_, img1);
	env->ReleaseStringUTFChars(img2_, img2);
	env->ReleaseStringUTFChars(videoPath_, videoPath);
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


/**
*	Cut an image into m*n patch
*	m*n  
*/
void cut_img(Mat src_img, int m, int n, vector<Mat> ceil_img,const char *path)
{
    int t = m * n;
    int height = src_img.rows;
    int width = src_img.cols;

    int ceil_height = height / m;
    int ceil_width = width / n;
    Mat roi_img;
    //String concatenation
    ostringstream oss;
    string  str, str1, str2;

    Point p1, p2;
    for (int i = 0; i<m; i++)
    {
        for (int j = 0; j<n; j++)
        {
            Rect rect(j*ceil_width, i*ceil_height, ceil_width, ceil_height);
            src_img(rect).copyTo(roi_img);
            ceil_img.push_back(roi_img);
			
			char name[512] = {0};
			sprintf(name, "%s/cut/%0d.jpg", path, i+j);

			imwrite(name, roi_img);
        }
    }
}

Mat splice_image(const string &img1, const string &img2, Mat frame1, Mat frame2){
double time = getTickCount();
	int64 app_start_time = getTickCount();

	//LOGD("match start time=%f\n", time);
if(!initialized_already)
{
	Mat pano;

#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif
	 // Check if have enough images
	// img_names.push_back(CUtil::jstringTostring(env,img1));
	// img_names.push_back(CUtil::jstringTostring(env,img2));
	img_names.push_back(img1);
	img_names.push_back(img2);

	num_images = static_cast<int>(img_names.size());
	if (num_images < 2){
		LOGD("Need more images---1");
		return empty;
	}

	//vector<Point> corners(num_images);
	//vector<UMat> masks_warped(num_images);
	//vector<UMat> images_warped(num_images);
	//vector<Size> sizes(num_images);
	//vector<UMat> masks(num_images);

	//corners = static_cast<Point>(num_images);
	//masks_warped = static_cast<UMat>(num_images);
	//images_warped = static_cast<UMat>(num_images);
	//sizes = static_cast<Size>(num_images);
	//masks = static_cast<UMat>(num_images);

	LOGD("Finding features...");

#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	initialized_already = true;
	Ptr<FeaturesFinder> finder;
	if (features_type == "surf"){
	#ifdef HAVE_OPENCV_XFEATURES2D
		if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
			finder = makePtr<SurfFeaturesFinderGpu>();
		else
	#endif
			finder = makePtr<SurfFeaturesFinder>();
	}
	else if (features_type == "orb"){
		finder = makePtr<OrbFeaturesFinder>();
	}
	else if (features_type == "sift") {
		finder = makePtr<SiftFeaturesFinder>();
	}
	else{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		LOGE("Unknown 2D features type");
		return empty;
	}

	for (int i = 0; i < num_images; ++i){
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty()){
		   // LOGD("Can't open image " << img_names[i]);
			LOGE("Can't open image");
			return empty;
		}
		if (work_megapix < 0){
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}else{
			if (!is_work_scale_set){
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		}
		if (!is_seam_scale_set){
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;

		// LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

		resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	//LOGD("Finding features, time=%f" , ((getTickCount() - t) / getTickFrequency()));

	LOGD("Pairwise matching");
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

	// LOGD("Pairwise matching, time=%f " ,((getTickCount() - t) / getTickFrequency()) );

	// Check if we should save matches graph
	if (save_graph){
		LOGD("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	//vector<Mat> img_subset;
   // vector<String> img_names_subset;
	//vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i){
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2){
		LOGE("Need more images---2");
		return empty;
	}

	Ptr<Estimator> estimator;
	if (estimator_type == "affine")
		estimator = makePtr<AffineBasedEstimator>();
	else
		estimator = makePtr<HomographyBasedEstimator>();


	if (!(*estimator)(features, pairwise_matches, cameras)){
		cout << "Homography estimation failed.\n";
		LOGE("Homography estimation failed.");
		return empty;
	}

	for (size_t i = 0; i < cameras.size(); ++i){
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		//LOGLN("Initial camera intrinsics #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj")
		adjuster = makePtr<detail::BundleAdjusterReproj>();
	else if (ba_cost_func == "ray")
		adjuster = makePtr<detail::BundleAdjusterRay>();
	else if (ba_cost_func == "affine")
		adjuster = makePtr<detail::BundleAdjusterAffinePartial>();
	else if (ba_cost_func == "no")
		adjuster = makePtr<NoBundleAdjuster>();
	else{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		LOGE("Unknown bundle adjustment cost function:");
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
	if (!(*adjuster)(features, pairwise_matches, cameras)){
		cout << "Camera parameters adjusting failed.\n";
		LOGE("Camera parameters adjusting failed");
		return empty;
	}

	// Find median focal length

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i){
		//LOGLN("Camera #" << indices[i] + 1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());

	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct){
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}

	LOGD("Warping images (auxiliary)... ");
#if ENABLE_LOG
	t = getTickCount();
#endif

	LOGD("cuda is support values :%d",cuda::getCudaEnabledDeviceCount());


	// Preapre images masks
	for (int i = 0; i < num_images; ++i){
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks


	#ifdef HAVE_OPENCV_CUDAWARPING
	if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0){
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

	if (!warper_creator){
		cout << "Can't create the following warper '" << warp_type << "'\n";
		LOGE("Can't create the following warper...");
		return empty;
	}

	 warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i){
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

	//  LOGD("Warping images, time=%f" ,((getTickCount() - t) / getTickFrequency()),);

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
		LOGE("Can't create the following seam finder...");
		return empty;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);

	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();

	LOGD("Compositing...");
#if ENABLE_LOG
	#endif

	Proto_frame1 = imread(img1);
	Proto_frame2 = imread(img2);
}
	//t = getTickCount();
	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;
	Ptr<Timelapser> timelapser;
	//double compose_seam_aspect = 1;
	double compose_work_aspect = 1;

	for (int img_idx = 0; img_idx < num_images; ++img_idx){
		//LOGD("Compositing image #%d" , indices[img_idx] + 1);
		if (!is_registration_finished)
		{
			switch (img_idx){
			case 0:
				full_img = Proto_frame1;
				break;
			case 1:
				full_img = Proto_frame2;
				is_registration_finished = true;
				break;
			}
		}
		else
		{
			switch (img_idx){
			case 0:
				full_img = frame1;
				break;
			case 1:
				full_img = frame2;
				break;
			}
		}
		// Read image and resize it if necessary
		//full_img = imread(img_names[img_idx]);
		if (!is_compose_scale_set){
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
			for (int i = 0; i < num_images; ++i){
				// Update intrinsics
				cameras[i].focal *= compose_work_aspect;
				cameras[i].ppx *= compose_work_aspect;
				cameras[i].ppy *= compose_work_aspect;

				// Update corner and size
				Size sz = full_img_sizes[i];
				if (std::abs(compose_scale - 1) > 1e-1){
					sz.width = cvRound(full_img_sizes[i].width * compose_scale);
					sz.height = cvRound(full_img_sizes[i].height * compose_scale);
				}

				Mat K;
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

		Mat K;
		cameras[img_idx].K().convertTo(K, CV_32F);

		// Warp the current image
		warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

		// Warp the current image mask
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

		// Compensate exposure
		//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();
		img.release();
		mask.release();

		dilate(masks_warped[img_idx], dilated_mask, Mat());
		resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
		mask_warped = seam_mask & mask_warped;

		if (!blender && !timelapse){
			blender = Blender::createDefault(blend_type, try_cuda);
			Size dst_sz = resultRoi(corners, sizes).size();
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_cuda);
			else if (blend_type == Blender::MULTI_BAND){
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				//LOGD("Multi-band blender, number of bands: " );
			}
			else if (blend_type == Blender::FEATHER){
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
				fb->setSharpness(1.f / blend_width);
				LOGD("Feather blender, sharpness: ");
			}
			blender->prepare(corners, sizes);
		}else if (!timelapser && timelapse){
			timelapser = Timelapser::createDefault(timelapse_type);
			timelapser->initialize(corners, sizes);
		}

		// Blend the current image
		if (timelapse){
			timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
			String fixedFileName;
			size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
			if (pos_s == String::npos){
				fixedFileName = "fixed_" + img_names[img_idx];
			}else{
				fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
			}
			imwrite(fixedFileName, timelapser->getDst());
		}else{
			blender->feed(img_warped_s, mask_warped, corners[img_idx]);
		}
	}

	if (!timelapse){
		Mat result, result_mask;
		blender->blend(result, result_mask);
		//char name[512] = {0};
		//sprintf(name, "%s/%0d.jpg", CUtil::jstringTostring(env,path).c_str(), step_turn);
		//LOGD("name = %s",name);
		//imwrite(name, result);
		//LOGD("splice image time=%f: ", ((getTickCount() - app_start_time) / getTickFrequency()));
		return result;
	}

	return empty;
};

//------------------------------------jni loaded----------------------------------------------------------
JNIEXPORT const char *classPathNameRx = "com/sensology/opencv/OpenUtil";

static JNINativeMethod methodsRx[] = { 
	{"playVideo","(Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;)D",(void*)playVideo},
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
