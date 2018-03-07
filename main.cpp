#include <opencv2/opencv.hpp>
#include <iostream>

double otsuFact;
double otsuThresh;
bool showSteps = true;

void updateOtsuThresh(cv::Mat gray, double thresh = 0, double scale = 0) {

	if (thresh == 0.0 && scale == 0.0) {

		cv::Mat a;
		otsuThresh = cv::threshold(
			gray, a, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU
		);

		//otsuThresh -= 40;
		otsuFact = 0.4;

		if (otsuThresh <= 0) {
			otsuThresh = 95;
			otsuFact = 0.34;
		}
	}
	else {
		otsuThresh = thresh;
		otsuFact = scale;
	}

}

cv::Mat canny(cv::Mat image) {

	cv::Mat edges;
	if (otsuFact == 0.0 || otsuThresh == 0.0) {
		updateOtsuThresh(image);
	}
	cv::Canny(image, edges, otsuThresh*otsuFact, otsuThresh);

	return edges;
}

cv::Vec3f getPupilHough (cv::Mat &image, cv::Mat grey_image) {

	std::vector<cv::Vec3f> circles;

	if (otsuFact == 0.0 || otsuThresh == 0.0) {
		updateOtsuThresh(grey_image);
	}

	// Apply the Hough Transform to find the iris
	cv::HoughCircles(grey_image, circles, CV_HOUGH_GRADIENT, 1, 99999, otsuThresh, otsuThresh*otsuFact, 0, grey_image.rows*0.4);

	// Draw the circles detected
	if (circles.size() > 0 ) {

		for (size_t i = 0; i < circles.size(); i++)
		{
			cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// circle cente/r
			cv::circle(image, center, 3, cv::Scalar(0, 255, 0), -1, 4, 0);
			// circle outline
			cv::circle(image, center, radius, cv::Scalar(0, 0, 255), 2, 4, 0);
		}
	return circles[0];
	}
	std::cout << "No circles found!!!";
	return circles[0];
}

std::vector<cv::Rect> detectEyes(cv::Mat &img, cv::Mat gray) {

	cv::CascadeClassifier eyes_cascade;
	if (!eyes_cascade.load("C:/opencv/sources/data/haarcascades_cuda/haarcascade_eye_tree_eyeglasses.xml")) { printf("--(!)Error loading\n"); };

	// A vector of Rect for storing bounding boxes for eyes.
	std::vector<cv::Rect> eyes;

	// Detect eyes. 
	equalizeHist(gray, gray);
	eyes_cascade.detectMultiScale(gray, eyes, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	for (size_t j = 0; j < eyes.size(); j++)
	{
		cv::Point center(eyes[j].x + eyes[j].width*0.5, eyes[j].y + eyes[j].height*0.5);
		double radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
		circle(img, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);

	}
	return eyes;

}

cv::Mat extractPupilForStream(cv::Mat img) {

	cv::resize(img, img, cv::Size(), 0.4, 0.4);
	
	cv::Mat gray = img.clone();
	cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
	updateOtsuThresh(gray);
	cv::Mat frame_eyes = img.clone();
	std::vector<cv::Rect> eyes = detectEyes(frame_eyes, gray);

	cv::Mat roi;
	cv::Mat roi_det;
	for (size_t j = 0; j < eyes.size(); j++) {
		roi = img(eyes[j]);
		roi_det = frame_eyes(eyes[j]);
	}

	cv::resize(frame_eyes, frame_eyes, cv::Size(), 0.5, 0.5);
	cv::imshow("Webcam", frame_eyes);
	cvWaitKey(0);
	
	cv::imshow("Webcam", roi_det);
	cvWaitKey(0);

	cv::Mat roi_gray = roi.clone();
	cv::cvtColor(roi_gray, roi_gray, cv::COLOR_BGR2GRAY);
	updateOtsuThresh(roi_gray);
	cv::Mat roi_gray2 = roi_gray.clone();
	canny(roi_gray2);
	cv::imshow("Canny", roi_gray2);
	GaussianBlur(roi_gray, roi_gray, cv::Size(9, 9), 4, 4);

	getPupilHough(roi, roi_gray);
	cv::imshow("Webcam", roi);
	cvWaitKey(0);

	return roi;

}

void findBestCircleTest(cv::Mat color, cv::Mat gray, cv::Mat canny) {

	cv::namedWindow("canny2"); cv::imshow("canny2", canny>0);

	std::vector<cv::Vec3f> circles;

	cv::HoughCircles(gray, circles, CV_HOUGH_GRADIENT, 1, 60, 200, 20, 0, 0);

	for (size_t i = 0; i < circles.size(); i++)
	{
		cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);
		cv::circle(color, center, 3, cv::Scalar(0, 255, 255), -1);
		cv::circle(color, center, radius, cv::Scalar(0, 0, 255), 1);
	}

	//compute distance transform:
	cv::Mat dt;
	cv::distanceTransform(255 - (canny>0), dt, CV_DIST_L2, 3);
	cv::namedWindow("distance transform"); cv::imshow("distance transform", dt / 255.0f);

	// test for semi-circles:
	float minInlierDist = 2.0f;
	for (size_t i = 0; i < circles.size(); i++)
	{
		// test inlier percentage:
		// sample the circle and check for distance to the next edge
		unsigned int counter = 0;
		unsigned int inlier = 0;

		cv::Point2f center((circles[i][0]), (circles[i][1]));
		float radius = (circles[i][2]);

		// maximal distance of inlier might depend on the size of the circle
		float maxInlierDist = radius / 25.0f;
		if (maxInlierDist<minInlierDist) maxInlierDist = minInlierDist;

		//TODO: maybe paramter incrementation might depend on circle size!
		for (float t = 0; t<2 * 3.14159265359f; t += 0.1f)
		{
			counter++;
			float cX = radius * cos(t) + circles[i][0];
			float cY = radius * sin(t) + circles[i][1];

			if (dt.at<float>(cY, cX) < maxInlierDist)
			{
				inlier++;
				cv::circle(color, cv::Point2i(cX, cY), 3, cv::Scalar(0, 255, 0));
			}
			else
				cv::circle(color, cv::Point2i(cX, cY), 3, cv::Scalar(255, 0, 0));
		}
		std::cout << 100.0f*(float)inlier / (float)counter << " % of a circle with radius " << radius << " detected" << std::endl;
	}

	cv::namedWindow("output"); cv::imshow("output", color);
	cv::imwrite("houghLinesComputed.png", color);

	cv::waitKey(0);
}

cv::Mat getIrisNorm(cv::Vec3f iris, cv::Mat img_gray, cv::Mat &dest) {

	cv::Vec3f pupil;
	pupil = iris ;
	int radius = iris[2];

	cv::Point center(cvRound(iris[0]), cvRound(iris[1]));
	int lastRad = 0;

	std::vector<cv::Point> points_on_circle;
	int xbound = std::min(img_gray.size().width - center.x, center.x);
	int ybound = std::min(img_gray.size().height - center.y, center.y);


	int boundry = std::min(xbound, ybound);
	double avgGlobBright = 0;
	int count = 1;
	int circleDensity = 360;
	//cv::Mat normIris(600, circleDensity+1, CV_8SC1);
	cv::Mat normIris;
	normIris = cv::Mat::zeros(cv::Size(361, img_gray.rows),img_gray.type());
	for (int i = radius + 10; i < boundry; i += 1) {
		cv::Size axes(i, i); // Ellipse axis dimensions, divided by 2, here the diameter will change every iteration
		cv::ellipse2Poly(center, axes, 0, 0, circleDensity, 1, points_on_circle);
		int avgBrightness = 0;
		cv::Mat test = img_gray.clone();
		test.at<uchar>(center) = 0;

		for (int j = 0; j < points_on_circle.size(); j++) {
			avgBrightness += img_gray.at<uchar>(points_on_circle.at(j));
			test.at<uchar>(points_on_circle.at(j)) = 0;
			normIris.at<uchar>(count - 1, j) = img_gray.at<uchar>(points_on_circle.at(j));
		}
		//cv::imshow("test", test);
		//cv::waitKey(0);
		avgBrightness /= circleDensity;
		if (count > 1) {
		avgGlobBright = avgGlobBright*(count-1);
		}
		avgGlobBright = (avgGlobBright + avgBrightness) / count;
		std::cout << "Radius  " << i << " current brightness: " << avgBrightness << " Avg brightness: " << avgGlobBright  << std::endl;
		if (avgBrightness > avgGlobBright*1.2 && count > radius && avgBrightness > 100) {
			pupil[2] = i;
			std::cout << "Radius  " << i << " with current brightness: " << avgBrightness << " reached threshold (20%)" << std::endl;
			cv::circle(dest, cv::Point(pupil[0], pupil[1]), pupil[2], cv::Scalar(0, 0, 255), 2, 4, 0);
			cv::Rect myROI(0, 0, circleDensity+1, count);
			normIris = normIris(myROI);
			return normIris;
		}
		else {
			lastRad = i;
			count++;
		}
		
	}
	
	pupil[2] = lastRad;
	std::cout << " Max Radius reached at " << lastRad;
	cv::circle(dest, cv::Point(pupil[0], pupil[1]), pupil[2], cv::Scalar(0, 0, 255), 2, 4, 0);
	cv::Rect myROI(0, 0, circleDensity+1, count);
	normIris = normIris(myROI);
	return normIris;
}

cv::Mat getEightOrdinal(cv::Mat irisNorm) {
	cv::Mat irisOrd;
	irisOrd = cv::Mat::zeros(cv::Size(irisNorm.cols, irisNorm.rows), irisNorm.type());

	for (int i = 1; i < irisNorm.rows - 1; i++) {
		for (int j = 1; j < irisNorm.cols - 1; j++) {
			int x = irisNorm.at <uchar>(i, j);
			irisOrd.at <uchar>(i - 1, j - 1) += (x > irisNorm.at <uchar>(i - 1, j - 1)) ? 1 : 0;
			irisOrd.at <uchar>(i - 1, j) += (x > irisNorm.at <uchar>(i - 1, j)) ? 1 : 0;
			irisOrd.at <uchar>(i - 1, j + 1) += (x > irisNorm.at <uchar>(i - 1, j + 1)) ? 1 : 0;
			irisOrd.at <uchar>(i, j + 1) += (x > irisNorm.at <uchar>(i, j + 1)) ? 1 : 0;
			irisOrd.at <uchar>(i + 1, j + 1) += (x > irisNorm.at <uchar>(i + 1, j + 1)) ? 1 : 0;
			irisOrd.at <uchar>(i + 1, j) += (x > irisNorm.at <uchar>(i + 1, j)) ? 1 : 0;
			irisOrd.at <uchar>(i + 1, j - 1) += (x > irisNorm.at <uchar>(i + 1, j - 1)) ? 1 : 0;
			irisOrd.at <uchar>(i, j - 1) += (x > irisNorm.at <uchar>(i, j - 1)) ? 1 : 0;
		}
}
	return irisOrd;
}

cv::Mat getRowOrdinal(cv::Mat irisNorm) {
	cv::Mat irisOrd;
	irisOrd = cv::Mat::zeros(cv::Size(irisNorm.cols, irisNorm.rows), irisNorm.type());

	for (int i = 0; i < irisNorm.rows - 2; i++) {
		for (int j = 0; j < irisNorm.cols - 1; j++) {
			int x = irisNorm.at <uchar>(i, j);
			irisOrd.at <uchar>(i, j) = (irisNorm.at <uchar>(i, j) > irisNorm.at <uchar>(i + 1, j)) ? 1 : 0;
		}
	}
	for (int j = 0; j < irisNorm.cols - 1; j++) {
		irisOrd.at <uchar>(irisNorm.rows - 1, j) = (irisNorm.at <uchar>(irisNorm.rows - 1, j) > irisNorm.at <uchar>( 1, j )) ? 1 : 0;
	}
	return irisOrd;
}

cv::Mat getColOrdinal(cv::Mat irisNorm) {
	cv::Mat irisOrd;
	irisOrd = cv::Mat::zeros(cv::Size(irisNorm.cols, irisNorm.rows), irisNorm.type());

	for (int i = 0; i < irisNorm.cols - 2; i++) {
		for (int j = 0; j < irisNorm.rows - 1; j++) {
			int x = irisNorm.at <uchar>(i, j);
			irisOrd.at <uchar>(j, i) = (irisNorm.at <uchar>(j, i) > irisNorm.at <uchar>(j , i + 1)) ? 1 : 0;
		}
	}
	for (int j = 0; j < irisNorm.rows - 1; j++) {
		irisOrd.at <uchar>(j , irisNorm.cols - 1) = (irisNorm.at <uchar>(j , irisNorm.cols - 1) > irisNorm.at <uchar>(j, 1)) ? 1 : 0;
	}
	return irisOrd;
}

void shiftMatrix(cv::Mat &mat, int shift) {
	cv::Mat org = mat.clone();

	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			if (j + shift < 0 || j + shift >= mat.cols) {
				mat.at <uchar>(i, j) = org.at <uchar>(i, (shift < 0) ? mat.cols-1 : 0);
			}
			else {
			mat.at <uchar>(i, j) = org.at <uchar>(i, j + shift);
			}
		}
	}


}

double matchOrdIris(cv::Mat t1, cv::Mat t2) {
	double match = 99999;
	double dist = 0;
	int rows = t1.rows;
	int cols = t1.cols;

	cv::Mat t1_org = t1.clone();

	for (int h = 0; h < rows*0.3; h++) {
		for (int w = 0; w < cols*0.3; w++) {

		}
	}

	for (int k = 1; k < 31; k++) {
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < cols; j++) {
				dist += abs( t1.at <uchar>(i, j) - t2.at <uchar>(i, j));
			}
		}
		if (match > dist) { 
			match = dist;
		}
		shiftMatrix(t1, (k < 10) ? 1 : -1);
	}
	
	return match;
}

void displayOrdIris(cv::Mat ord, int max, std::string s) {

	for (int i = 0; i < ord.rows; i++) {
		for (int j = 0; j < ord.cols; j++) {
			double d = ord.at <uchar>(i, j) ;
			ord.at <uchar>(i, j) = d * (255 / max);
			ord.at <uchar>(i, j) = (int) ord.at <uchar>(i, j);
		}
	}
	if (showSteps) {
	cv::imshow("Ordinal Iris "+s, ord);
	cv::imwrite(s+".png", ord);
	}
}

void processImage(cv::Mat img, cv::Mat &ordNormIris, std::string dat, std::string alg) {

	cv::Mat gray = img.clone();

	cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
	cv::Mat trunc;
		
	cv::threshold(gray, trunc, 45, 255, cv::THRESH_TRUNC);

	// for img3:
	// cv::threshold(gray, trunc, 60, 255, cv::THRESH_TRUNC);

	// if(center 8pixel is below 240 in average/no circle or wrong cirle found:
	// cv::threshold(gray, trunc, 30, 255, cv::THRESH_TRUNC);
	// cv::threshold(trunc, trunc, 25, 255, cv::THRESH_BINARY);
	// test(src, trunc, canny(trunc));

	cv::Mat irisCirlce = img.clone();

	std::vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);
	
	if (showSteps){
		cv::imshow("gray", gray);
		cv::imshow("trunc", trunc);
		cv::imwrite(dat+"-gray.png", gray);
		cv::imwrite(dat+"-trunc.png", trunc);
		cv::waitKey(0);

		}
	
	cv::Vec3f iris = getPupilHough(irisCirlce, trunc);
	cv::Mat normIris = getIrisNorm(iris, gray, irisCirlce);
	cv::resize(normIris, normIris, cv::Size(361, 128), 0, 0, CV_INTER_CUBIC);

	cv::Mat ordImg;
	if (alg.compare("lbp")) {
		ordImg = getEightOrdinal(normIris);
		if (showSteps) {
			displayOrdIris(ordImg, 8, dat + "-ordinalIris_LBP");
		}
	}
	else if (alg.compare("vertical")) {
		ordImg = getColOrdinal(normIris);
		if (showSteps) {
			displayOrdIris(ordImg, 1, dat + "-ordinalIris_vertical");
		}
	}
	else {
		ordImg = getRowOrdinal(normIris);
		if (showSteps) {
			displayOrdIris(ordImg, 1, dat + "-ordinalIris");
		}
	}

	if (showSteps) {
	cv::imshow(dat + "-hough", irisCirlce);
	cv::imshow(dat + "-normIris", normIris);

	cv::imwrite(dat+"-hough.png", irisCirlce);
	cv::imwrite(dat+"-normIris.png", normIris);

	cv::waitKey(0);
	}
	ordNormIris = ordImg.clone();

}

void matchRes(cv::Mat &img1, cv::Mat &img2) {
	double res1 = img1.rows*img1.cols;
	double res2 = img2.rows*img2.cols;
	bool firstSmaler = (res1 > res2) ? false : true;

	double scale;
	
	if (res1 > res2) {
		scale = sqrt((res2 / res1));
		cv::resize(img1, img1, cv::Size(), scale, scale);
	}
	else {
		scale = sqrt((res1 / res2));
		cv::resize(img2, img2, cv::Size(), scale, scale);
	}

}

void convertImage(std::string path, std::string img_file, std::string img_file2, std::string alg) {

	std::string image_path1 = path+ img_file;
	std::string image_path2 = path +img_file2;
	showSteps = true;

	cv::Mat src1 = cv::imread(image_path1);
	cv::Mat irisOrd1;
	cv::Mat src2 = cv::imread(image_path2);
	cv::Mat irisOrd2;

	cv::resize(src1, src1, cv::Size(), 1, 1);
	matchRes(src1, src2);

	if (showSteps) {
	cv::imshow("img1", src1);
	cv::imshow("img2", src2);
	cvWaitKey(0);
	}

	processImage(src1, irisOrd1, img_file, alg);
	processImage(src2, irisOrd2, img_file2, alg);

	// 15,000~
	// 20,000~
	// std::cout << std::endl << "Matching:" << matchOrdIris(irisOrd1, irisOrd2) << std::endl;

	cv::waitKey(0);
}

void stream() {

	int fps = 30;
	cv::VideoCapture vid(0);
	// vid.open("https://192.168.137.112:6060/video");

	cv::Mat frame;


	bool stream_on = true;


	while (vid.read(frame)) {
		
		if (cv::waitKey(1000 / fps) >= 0) {
			break;
		}

		//cv::resize(frame, frame, cv::Size(), 0.7, 0.7);
		cv::Mat gray = frame.clone();
		
		cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);

		updateOtsuThresh(gray);

		GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
		detectEyes(frame, gray);
		//hough(frame, gray);
		cv::imshow("Webcam", frame);


	}

	vid.read(frame);
	cv::Mat gray = frame.clone();
	cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
	updateOtsuThresh(gray);
	cv::Mat frame_eyes = frame.clone();
	std::vector<cv::Rect> eyes = detectEyes(frame_eyes, gray);
	cv::Mat roi;
	cv::Mat roi_det;
	for (size_t j = 0; j < eyes.size(); j++) {
		roi = frame(eyes[j]);
		roi_det = frame_eyes(eyes[j]);
	}
	cv::imshow("Webcam", roi_det);
	cvWaitKey(0);

	cv::Mat roi_gray = roi.clone();
	cv::cvtColor(roi_gray, roi_gray, cv::COLOR_BGR2GRAY);
	
	GaussianBlur(roi_gray, roi_gray, cv::Size(9, 9), 2, 2);
	updateOtsuThresh(roi_gray);
	
	getPupilHough(roi, roi_gray);
	cv::imshow("Webcam", roi);
	cvWaitKey(0);



}

void stream2(std::string alg) {

	int fps = 30;
	cv::VideoCapture vid(0);
	cv::Mat frame;

	while (vid.read(frame)) {
		if (cv::waitKey(1000 / fps) >= 0) { break; }

		cv::Mat gray;
		cv::resize(frame, gray, cv::Size(), 1, 1);
		cv::cvtColor(gray, gray, cv::COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
		//cv::imshow("webcam", frame);

		if (alg.compare( "lbp") ) {
			cv::Mat ordImg = getEightOrdinal(gray);
			displayOrdIris(ordImg, 8, "ord");
		}
		else if (alg.compare("vertical")) {
			cv::Mat ordImg = getColOrdinal(gray);
			displayOrdIris(ordImg, 1, "ord");
		}
		else {
			cv::Mat ordImg = getRowOrdinal(gray);
			displayOrdIris(ordImg, 1, "ord");
		}
	}
}

int main() {

	// Takes an eye from a stream, was abandoned after results show image quality is too low; Uses Haar-like features to detect eyes
	// pressing key will crop to one of the eyes
	// stream();

	// Shows intermediate steps, press key to continue when images show
	showSteps = true;

	std::string path = "C:/Users/Philipp/Desktop/#Uni SO/2. Semester/Advanced Computer Vision/week 5/IrisOrdinal/";
	// "vertical", "horizontal" or "LBP" allowed
	// saves output in project path
	convertImage(path, "img1.jpg", "img2.jpg", "lbp");
	
	// Transforms the input stream into ordinal representation, either "vertical", "horizontal" or "LBP" are allowed
	// stream2("lbp");
	
	return 1;
}
