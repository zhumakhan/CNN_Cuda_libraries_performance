#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <iostream>
#include <cstring>
#include <cuda_profiler_api.h>


using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        // if (severity <= Severity::kWARNING)
        // std::cout << msg << std::endl;
    }
} logger;


int main(){
    IBuilder * builder = createInferBuilder(logger);
    std::cout<<"Float16 support: "<<builder->platformHasFastFp16()<<std::endl;
    std::cout<<"Int8 support: "<<builder->platformHasFastInt8()<<std::endl;
    
    INetworkDefinition * network = builder->createNetworkV2(0);
    IBuilderConfig * config = builder->createBuilderConfig();
    
    // config->setFlag(BuilderFlag::kDEBUG);
    builder->setMaxBatchSize(32);
    // config->setFlag(BuilderFlag::kFP16);
    
    ITensor * input = network->addInput("in", DataType::kFLOAT, Dims4{1,384,26,26});

    
    //conv layer
    float* weights = (float*)malloc(256*384*3*3*sizeof(float));
    float* bias = (float*)malloc(256*sizeof(float));

    Weights conv_w{DataType::kFLOAT, weights, 256*384*3*3};
    Weights conv_b{DataType::kFLOAT, bias, 256};

    IConvolutionLayer * conv_layer = network->addConvolutionNd(
        *input,
        256,
        Dims2{3,3},
        conv_w,
        conv_b
    );
    conv_layer->setStride(DimsHW{1,1});
    conv_layer->setPadding(DimsHW{0,0});
    conv_layer->setNbGroups(1);
    conv_layer->setName("Conv3");

    input = conv_layer->getOutput(0);
    input->setName("out");
    network->markOutput(*input);


    //build an inference engine from the network
    // IHostMemory * serialized_engine = builder->buildSerializedNetwork(*network, *config);
    ICudaEngine * engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext * context = engine->createExecutionContext();
    
    std::cout<<"input/output numbers: "<<engine->getNbBindings()<<std::endl;
    int buf_input_idx = engine->getBindingIndex("in");
    int buf_output_idx = engine->getBindingIndex("out");
    std::cout<<"Number of layers in network: "<<network->getNbLayers()<<std::endl;
    
    //Memory management
    void * buffer[2];//binding memory space of input and output

    cudaMalloc(&buffer[0], 256*26*26*sizeof(float));
    cudaMalloc(&buffer[1], 256*26*26*sizeof(float));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaProfilerStart();
    for(int i = 0; i < 10; i++){
        context->enqueue(1, buffer, stream, nullptr);
    }
    cudaProfilerStop();
    
    free(weights);
    free(bias);
    
    cudaFree(buffer[0]);
    cudaFree(buffer[1]);
    
    context->destroy();
    engine->destroy();

	return 0;

}