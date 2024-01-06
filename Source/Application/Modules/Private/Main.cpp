// Author: Lucas Vilas-Boas
// Year : 2023
// Repo : https://github.com/lucoiso/opencv-experiment

#include <opencv2/opencv.hpp>

constexpr std::uint32_t ESC_KEY_CODE = 27U;
constexpr char const *const WINDOW_NAME = "OpenCV Experimental Classification Application";

int main([[maybe_unused]] int const Argc, [[maybe_unused]] char *const Argv[])
{
    try
    {
        namedWindow(WINDOW_NAME, cv::WINDOW_AUTOSIZE);

        constexpr std::array Datasets {"haarcascade_frontalface_default.xml"};

        std::vector<cv::CascadeClassifier> Classifiers;
        Classifiers.resize(std::size(Datasets));

        for (std::uint8_t Index = 0U; Index < static_cast<std::uint8_t>(std::size(Datasets)); ++Index)
        {
            if (!Classifiers[Index].load(Datasets[Index]))
            {
                throw std::runtime_error("failed to load data");
            }
        }

        if (std::empty(Classifiers))
        {
            throw std::runtime_error("no classifiers loaded");
        }

        cv::VideoCapture Camera(0, cv::VideoCaptureAPIs::CAP_ANY);
        if (!Camera.isOpened())
        {
            throw std::runtime_error("failed to open camera");
        }

        cv::Mat CurrentFrame;
        cv::Mat GrayScaleImage;

        auto const ProcessClassifier = [&CurrentFrame, &GrayScaleImage](cv::CascadeClassifier &Classifier, cv::Scalar const &Color)
        {
            std::vector<cv::Rect> Objects;
            Classifier.detectMultiScale(GrayScaleImage, Objects, 1.3, 2, cv::CASCADE_SCALE_IMAGE, cv::Size(8, 8));

            for (cv::Rect const &ObjectIter: Objects)
            {
                rectangle(CurrentFrame, ObjectIter, Color);
            }
        };

        while (Camera.read(CurrentFrame) && !std::empty(CurrentFrame))
        {
            cvtColor(CurrentFrame, GrayScaleImage, cv::COLOR_BGR2GRAY);
            equalizeHist(GrayScaleImage, GrayScaleImage);

            if (cv::Ptr<cv::CLAHE> const Clahe = createCLAHE(2.0, cv::Size(8, 8)))
            {
                Clahe->apply(GrayScaleImage, GrayScaleImage);
            }

            for (cv::CascadeClassifier &ClassifierIter: Classifiers)
            {
                ProcessClassifier(ClassifierIter, cv::Scalar(0, 255, 0));
            }

            imshow(WINDOW_NAME, CurrentFrame);

            if (cv::pollKey() == ESC_KEY_CODE || getWindowProperty(WINDOW_NAME, cv::WND_PROP_VISIBLE) == 0.0)
            {
                break;
            }
        }

        return EXIT_SUCCESS;
    }
    catch (std::exception const &Exception)
    {
        std::cerr << Exception.what() << std::endl;
    }

    return EXIT_FAILURE;
}
