// Compare the C++ preprocessing and inference with the probabilities that Python computed.
//
// The test also checks itself: six plausible porting errors in the preprocessing must each exceed
// the tolerance, otherwise the tolerance is too loose to catch them.
//
// Usage: test_parity FIXTURE_DIR, where the directory comes from `sdi parity-fixture`.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "classifier.hpp"

namespace {

// The correct preprocessing deviates by about 7e-8 on the fixture (Ubuntu 24.04, OpenCV 4.6).
// The tolerance leaves two orders of magnitude for other platforms. With 1e-4, resizing in float
// instead of 8 bit passed on the fixture (5.4e-5) although it shifts probabilities on real images
// by up to 0.05.
constexpr double kTolerance = 1e-5;

// The numbers of the "probabilities" array; the fixture is written by our own code, so a full
// JSON parser is not needed.
std::vector<double> expected_probabilities(const std::string& path) {
  std::ifstream file(path);
  std::stringstream buffer;
  buffer << file.rdbuf();
  const std::string text = buffer.str();
  std::vector<double> values;
  auto key = text.find("\"probabilities\"");
  if (key == std::string::npos) return values;
  const char* cursor = text.c_str() + text.find('[', key) + 1;
  while (*cursor != ']' && *cursor != '\0') {
    char* end = nullptr;
    double value = std::strtod(cursor, &end);
    if (end == cursor) {
      ++cursor;
      continue;
    }
    values.push_back(value);
    cursor = end;
  }
  return values;
}

double largest_difference(const std::vector<float>& actual, const std::vector<double>& expected) {
  double worst = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    worst = std::max(worst, std::abs(actual[i] - expected[i]));
  }
  return worst;
}

// Resize the 8-bit image and scale it to [0, 1], as sdi::preprocess does.
cv::Mat resized(const cv::Mat& gray, int interpolation) {
  cv::Mat small, scaled;
  cv::resize(gray, small, cv::Size(sdi::kSize, sdi::kSize), 0, 0, interpolation);
  small.convertTo(scaled, CV_32F, 1.0 / 255.0);
  return scaled;
}

cv::Mat normalised(const cv::Mat& scaled, float mean = sdi::kMean, float std = sdi::kStd) {
  return (scaled - mean) / std;
}

using Preprocess = std::function<cv::Mat(const cv::Mat&)>;

// Each returns a normalised 224 x 224 float image with one porting error.
const std::vector<std::pair<std::string, Preprocess>> kWrongPreprocessing = {
    {"nearest", [](const cv::Mat& g) { return normalised(resized(g, cv::INTER_NEAREST)); }},
    {"area", [](const cv::Mat& g) { return normalised(resized(g, cv::INTER_AREA)); }},
    // The ImageNet constants of the red channel instead of their mean over the three channels.
    {"red_channel_constants",
     [](const cv::Mat& g) { return normalised(resized(g, cv::INTER_LINEAR), 0.485F, 0.229F); }},
    {"no_normalisation", [](const cv::Mat& g) { return resized(g, cv::INTER_LINEAR); }},
    {"transposed",
     [](const cv::Mat& g) { return cv::Mat(normalised(resized(g, cv::INTER_LINEAR)).t()); }},
    {"float_resize",
     [](const cv::Mat& g) {
       cv::Mat scaled, small;
       g.convertTo(scaled, CV_32F, 1.0 / 255.0);
       cv::resize(scaled, small, cv::Size(sdi::kSize, sdi::kSize), 0, 0, cv::INTER_LINEAR);
       return normalised(small);
     }},
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s FIXTURE_DIR\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const auto expected = expected_probabilities(dir + "/expected.json");
  sdi::Classifier classifier(dir + "/model.onnx");
  const cv::Mat image = cv::imread(dir + "/image.png", cv::IMREAD_GRAYSCALE);
  const auto actual = classifier.probabilities(image);
  if (expected.size() != sdi::kClasses.size() || actual.size() != expected.size()) {
    std::fprintf(stderr, "expected %zu probabilities, got %zu from JSON and %zu from C++\n",
                 sdi::kClasses.size(), expected.size(), actual.size());
    return 1;
  }
  const double worst = largest_difference(actual, expected);
  std::printf("largest difference %.2e (tolerance %.0e)\n", worst, kTolerance);
  bool ok = worst <= kTolerance;

  cv::dnn::Net net = cv::dnn::readNetFromONNX(dir + "/model.onnx");
  for (const auto& [name, preprocess] : kWrongPreprocessing) {
    net.setInput(cv::dnn::blobFromImage(preprocess(image)), "image");
    const cv::Mat out = net.forward("probabilities");
    const double difference =
        largest_difference(std::vector<float>(out.begin<float>(), out.end<float>()), expected);
    const bool caught = difference > kTolerance;
    std::printf("%-22s %.2e %s\n", name.c_str(), difference, caught ? "caught" : "NOT CAUGHT");
    ok = ok && caught;
  }
  return ok ? 0 : 1;
}
