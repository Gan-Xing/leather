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

    // 加载图像（假设图像已经预处理完成）
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        std::cerr << "Failed to load image." << std::endl;
        return 1;
    }

    // 确保图像是浮点类型，进行归一化
    img.convertTo(img, CV_32F, 1.0 / 255.0);  // 将像素值范围从[0, 255]映射到[0, 1]
    cv::subtract(img, cv::Scalar(0.485, 0.456, 0.406), img, cv::noArray(), -1);
    cv::divide(img, cv::Scalar(0.229, 0.224, 0.225), img, 1, -1);

    std::vector<float> img_data((float*)img.datastart, (float*)img.dataend);

    std::vector<int64_t> input_tensor_shape = {1, img.channels(), img.rows, img.cols};
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
    std::cout << "Predicted Class Index: " << result_idx << std::endl; // 只输出结果

    return 0;
}
