#include "utils.hpp"            // custom util function
#include "logging.hpp"          // Nvidia logger
#include "calibrator.hpp"       // for ptq
#include "parserOnnxConfig.h"   // for onnx-parsing

using namespace nvinfer1;
sample::Logger gLogger;

static const int maxBatchSize = 1;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 1000;
static const int precision_mode = 8;        // fp32 : 32, fp16 : 16, int8(ptq) : 8
const char* INPUT_BLOB_NAME = "input";      // use same input name with onnx model
const char* OUTPUT_BLOB_NAME = "output";    // use same output name with onnx model
const char* engineFileName = "resnet18";    // model name
const char* onnx_file = "../../PyTorch/model/resnet18_cuda_trt.onnx"; // onnx model file path
bool ONNX_MODEL = true;                     // true : from ONNX , false : using TRT API
bool serialize = false;                     // force serialize flag (IF true, recreate the engine file unconditionally)
uint64_t iter_count = 10000;                // the number of test iterations

// Creat the engine using onnx.
void createEngineFromOnnx(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engine_file_path);

// Creat the engine using only the API and not any parser.
void createEngineUsingAPI(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName);

int main()
{
    char engine_file_path[256];
    sprintf(engine_file_path, "../Engine/%s_%d.engine", engineFileName, precision_mode);

    /*
    /! 1) Create engine file 
    /! If force serialize flag is true, recreate unconditionally
    /! If force serialize flag is false, engine file is not created if engine file exist.
    /!                                   create the engine file if engine file doesn't exist.
    */
    bool exist_engine = false;
    if ((access(engine_file_path, 0) != -1)) {
        exist_engine = true;
    }

    if (!((serialize == false)/*Force Serialize flag*/ && (exist_engine == true) /*Whether the resnet18.engine file exists*/)) {
        std::cout << "===== Create Engine file =====" << std::endl << std::endl;

        IBuilder* builder = createInferBuilder(gLogger);
        if (!builder){
            std::string msg("failed to make builder");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }

        IBuilderConfig* config = builder->createBuilderConfig();
        if (!config) {
            std::string msg("failed to make config");
            gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
            exit(EXIT_FAILURE);
        }
        
        if (ONNX_MODEL) {
            // ***  create tensorrt model from ONNX Model ***
            createEngineFromOnnx(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path);
        }
        else {
            // *** create tensorrt model using by TensorRT API ***
            createEngineUsingAPI(maxBatchSize, builder, config, DataType::kFLOAT, engine_file_path); 
        }

        builder->destroy();
        config->destroy();
        std::cout << "===== Create Engine file =====" << std::endl << std::endl; 
    }

    // 2) load engine file
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::cout << "===== Engine file load =====" << std::endl << std::endl;
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        file.read(trtModelStream, size);
        file.close();
    }
    else {
        std::cout << "[ERROR] Engine file load error" << std::endl;
    }

    // 3) deserialize TensorRT Engine from file
    std::cout << "===== Engine file deserialize =====" << std::endl << std::endl;
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    IExecutionContext* context = engine->createExecutionContext();
    delete[] trtModelStream;

    void* buffers[2];
    const int inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);

    // Allocate GPU memory space for input and output
    CHECK(cudaMalloc(&buffers[inputIndex], maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float)));

    // 4) prepare input data
    std::string img_dir = "../../data/";
    std::vector<std::string> file_names;
    if (SearchFile(img_dir.c_str(), file_names) < 0) { // load input data
        std::cerr << "[ERROR] Data search error" << std::endl;
    }
    else {
        std::cout << "Total number of images : " << file_names.size() << std::endl << std::endl;
    }
    cv::Mat img(INPUT_H, INPUT_W, CV_8UC3);
    cv::Mat ori_img;
    std::vector<uint8_t> input_i8(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> input(maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
    std::vector<float> outputs(OUTPUT_SIZE);
    for (int idx = 0; idx < maxBatchSize; idx++) { // mat -> vector<uint8_t>
        cv::Mat ori_img = cv::imread(file_names[idx]);
        //cv::resize(ori_img, img, img.size());
        memcpy(input_i8.data(), ori_img.data, maxBatchSize * INPUT_H * INPUT_W * INPUT_C);
        Preprocess(input, input_i8, maxBatchSize , INPUT_C, INPUT_H, INPUT_W);
    }
    std::cout << "===== input load done =====" << std::endl << std::endl;

    uint64_t dur_time = 0;


    // CUDA stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // Warm-up
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context->enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // 5) Inference
    for (int i = 0; i < iter_count; i++) {
        auto start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        CHECK(cudaMemcpyAsync(buffers[inputIndex], input.data(), maxBatchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
        context->enqueueV2(buffers, stream, nullptr);
        CHECK(cudaMemcpyAsync(outputs.data(), buffers[outputIndex], maxBatchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);

        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - start;
        dur_time += dur;
        //std::cout << dur << " milliseconds" << std::endl;
    }
    dur_time /= 1000.f; //microseconds -> milliseconds

    // 6) Print Results
    std::cout << "==================================================" << std::endl;
    std::cout << "Model : " << engineFileName << ", Precision : " << precision_mode << std::endl;
    std::cout << iter_count << " th Iteration" << std::endl;
    std::cout << "Total duration time with data transfer : " << dur_time << " [milliseconds]" << std::endl;
    std::cout << "Avg duration time with data transfer : " << dur_time / (float)iter_count << " [milliseconds]" << std::endl;
    std::cout << "FPS : " << 1000.f / (dur_time / (float)iter_count) << " [frame/sec]" << std::endl;
    int max_index = max_element(outputs.begin(), outputs.end()) - outputs.begin();
    std::cout << "Index : " << max_index << ", Probability : " << outputs[max_index] << std::endl;
    std::cout << "Class Name : " << class_names[max_index] << std::endl;
    std::cout << "==================================================" << std::endl;

    // Release stream and buffers ...
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}

// Creat the engine using onnx.
void createEngineFromOnnx(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engine_file_path)
{
    std::cout << "==== model build start ====" << std::endl << std::endl;

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
    if (!network) {
        std::string msg("failed to make network");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    if (!parser->parseFromFile(onnx_file, (int)nvinfer1::ILogger::Severity::kINFO)) {
        std::string msg("failed to parse onnx file");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }

    // Build engine
    //builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30); // 30:1GB, 29:512MB
    if (precision_mode == 16) {
        std::cout << "==== precision f16 ====" << std::endl << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == 8) {
        std::cout << "==== precision int8 ====" << std::endl << std::endl;
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_C, INPUT_W, INPUT_H, "../../calib_data/", "../Engine/resnet18_i8_calib.table", INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }
    else {
        std::cout << "==== precision f32 ====" << std::endl << std::endl;
    }

    bool dlaflag = false;
    int32_t dlaCore = builder->getNbDLACores();
    bool allowGPUFallback = true;
    std::cout << "the number of DLA engines available to this builder :: " << dlaCore << std::endl << std::endl;
    if (dlaCore >= 0 && dlaflag) {
        if (builder->getNbDLACores() == 0) {
            std::cerr << "Trying to use DLA core on a platform that doesn't have any DLA cores"
                << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allowGPUFallback) {
            config->setFlag(BuilderFlag::kGPU_FALLBACK);
        }
        if (!config->getFlag(BuilderFlag::kINT8)) {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            config->setFlag(BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(DeviceType::kDLA);
        config->setDLACore(dlaCore);
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;

    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    if (!engine) {
        std::string msg("failed to make engine");
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, msg.c_str());
        exit(EXIT_FAILURE);
    }
    std::cout << "==== model build done ====" << std::endl << std::endl;

    std::cout << "==== model selialize start ====" << std::endl << std::endl;
    std::ofstream p(engine_file_path, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl << std::endl;
    }
    p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    std::cout << "==== model selialize done ====" << std::endl << std::endl;

    engine->destroy();
    network->destroy();
    p.close();
}

IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{ DataType::kFLOAT, scval, len };

    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{ DataType::kFLOAT, shval, len };

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{ DataType::kFLOAT, pval, len };

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

IActivationLayer* basicBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, int inch, int outch, int stride, std::string lname) {
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ 3, 3 }, weightMap[lname + "conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ stride, stride });
    conv1->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{ 3, 3 }, weightMap[lname + "conv2.weight"], emptywts);
    assert(conv2);
    conv2->setPaddingNd(DimsHW{ 1, 1 });

    IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

    IElementWiseLayer* ew1;
    if (inch != outch) {
        IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{ 1, 1 }, weightMap[lname + "downsample.0.weight"], emptywts);
        assert(conv3);
        conv3->setStrideNd(DimsHW{ stride, stride });
        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
        ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    else {
        ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
    }
    IActivationLayer* relu2 = network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
    assert(relu2);
    return relu2;
}

// Creat the engine using only the API and not any parser.
void createEngineUsingAPI(int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, char* engineFileName)
{
    std::cout << "==== model build start ====" << std::endl << std::endl;
    INetworkDefinition* network = builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)); //explicit batch mode [N,C,H,W]

    std::map<std::string, Weights> weightMap = loadWeights("../../PyTorch/model/resnet18.wts");
    Weights emptywts{ DataType::kFLOAT, nullptr, 0 };

    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims4{ maxBatchSize,  INPUT_C, INPUT_H, INPUT_W });
    assert(data);

    IConvolutionLayer* conv1 = network->addConvolutionNd(*data, 64, DimsHW{ 7, 7 }, weightMap["conv1.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{ 2, 2 });
    conv1->setPaddingNd(DimsHW{ 3, 3 });

    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "bn1", 1e-5);

    IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{ 3, 3 });
    assert(pool1);
    pool1->setStrideNd(DimsHW{ 2, 2 });
    pool1->setPaddingNd(DimsHW{ 1, 1 });

    IActivationLayer* relu2 = basicBlock(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "layer1.0.");
    IActivationLayer* relu3 = basicBlock(network, weightMap, *relu2->getOutput(0), 64, 64, 1, "layer1.1.");

    IActivationLayer* relu4 = basicBlock(network, weightMap, *relu3->getOutput(0), 64, 128, 2, "layer2.0.");
    IActivationLayer* relu5 = basicBlock(network, weightMap, *relu4->getOutput(0), 128, 128, 1, "layer2.1.");

    IActivationLayer* relu6 = basicBlock(network, weightMap, *relu5->getOutput(0), 128, 256, 2, "layer3.0.");
    IActivationLayer* relu7 = basicBlock(network, weightMap, *relu6->getOutput(0), 256, 256, 1, "layer3.1.");

    IActivationLayer* relu8 = basicBlock(network, weightMap, *relu7->getOutput(0), 256, 512, 2, "layer4.0.");
    IActivationLayer* relu9 = basicBlock(network, weightMap, *relu8->getOutput(0), 512, 512, 1, "layer4.1.");

    IPoolingLayer* pool2 = network->addPoolingNd(*relu9->getOutput(0), PoolingType::kAVERAGE, DimsHW{ 7, 7 });
    assert(pool2);
    pool2->setStrideNd(DimsHW{ 1, 1 });

    IFullyConnectedLayer* fc1 = network->addFullyConnected(*pool2->getOutput(0), 1000, weightMap["fc.weight"], weightMap["fc.bias"]);
    assert(fc1);

    fc1->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*fc1->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(1ULL << 30); // 512MB

    if (precision_mode == 16) {
        std::cout << "==== precision f16 ====" << std::endl << std::endl;
        config->setFlag(BuilderFlag::kFP16);
    }
    else if (precision_mode == 8) {
        std::cout << "==== precision int8 ====" << std::endl << std::endl;
        std::cout << "Your platform support int8: " << builder->platformHasFastInt8() << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(maxBatchSize, INPUT_C, INPUT_W, INPUT_H, "../../calib_data/", "../Engine/resnet18_i8_calib.table", INPUT_BLOB_NAME);
        config->setInt8Calibrator(calibrator);
    }
    else {
        std::cout << "==== precision f32 ====" << std::endl << std::endl;
    }

    std::cout << "Building engine, please wait for a while..." << std::endl;
    IHostMemory* engine = builder->buildSerializedNetwork(*network, *config);
    std::cout << "==== model build done ====" << std::endl << std::endl;

    std::cout << "==== model selialize start ====" << std::endl << std::endl;
    std::ofstream p(engineFileName, std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl << std::endl;
    }
    p.write(reinterpret_cast<const char*>(engine->data()), engine->size());
    std::cout << "==== model selialize done ====" << std::endl << std::endl;
    engine->destroy();
    network->destroy();
    p.close();
    // Release host memory
    for (auto& mem : weightMap)
    {
        free((void*)(mem.second.values));
    }
}
