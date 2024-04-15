#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <iostream>

// 函数：应用Softmax
std::vector<float> softmax(const std::vector<float>& z) {
    float max_elem = *std::max_element(z.begin(), z.end());
    std::vector<float> exp_z(z.size());
    float sum = 0.0;

    for (size_t i = 0; i < z.size(); i++) {
        exp_z[i] = exp(z[i] - max_elem);
        sum += exp_z[i];
    }

    for (size_t i = 0; i < z.size(); i++) {
        exp_z[i] /= sum;
    }

    return exp_z;
}

// 函数：执行中心裁剪
cv::Mat centerCrop(const cv::Mat &img, const int cropSize) {
    const int offsetW = (img.cols - cropSize) / 2;
    const int offsetH = (img.rows - cropSize) / 2;
    const cv::Rect roi(offsetW, offsetH, cropSize, cropSize);
    return img(roi);
}

// 主函数
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }

    const char* image_path = argv[1];
    const char* model_path = "../../models/resnet18_leather_bestjit.onnx"; // 相对于cppinference目录


    // 初始化ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // 加载模型
    Ort::Session session(env, model_path, session_options);

    // 加载和预处理图像
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // 先执行中心裁剪至512x512
    cv::Mat img_cropped = centerCrop(img, 512);

    // 然后调整大小至256x256
    cv::resize(img_cropped, img_cropped, cv::Size(256, 256));

    // 再次执行中心裁剪至224x224
    img_cropped = centerCrop(img_cropped, 224);

    // 数据转换为Tensor
    img_cropped.convertTo(img_cropped, CV_32F, 1.0 / 255.0);
    cv::subtract(img_cropped, cv::Scalar(0.485, 0.456, 0.406), img_cropped, cv::noArray(), -1);
    cv::divide(img_cropped, cv::Scalar(0.229, 0.224, 0.225), img_cropped, 1, -1);

    std::vector<float> img_data((float*)img_cropped.datastart, (float*)img_cropped.dataend);

    std::vector<int64_t> input_tensor_shape = {1, 3, 224, 224};
    std::vector<float> output_tensor_values(8); // 假设模型有8个输出
    std::vector<int64_t> output_tensor_shape = {1, 8};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, img_data.data(), img_data.size(), input_tensor_shape.data(), input_tensor_shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info, output_tensor_values.data(), output_tensor_values.size(), output_tensor_shape.data(), output_tensor_shape.size());

    // 执行推理
    const char* input_names[] = {"x.1"};  // 使用模型中定义的确切输入名称
    const char* output_names[] = {"171"};  // 使用模型中定义的确切输出名称
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

    // 应用Softmax
    std::vector<float> softmax_output = softmax(output_tensor_values);
    int result_idx = std::distance(softmax_output.begin(), std::max_element(softmax_output.begin(), softmax_output.end()));

    // 输出预测结果
    std::cout << result_idx << std::endl; // 只输出结果

    return 0;
}
