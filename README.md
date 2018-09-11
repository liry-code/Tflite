# Tflite
c++ implement tflite android demo

## load model 
errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
modelHandle = createModel(modelPath, errorHandle);
interpreterHandle = createInterpreter(modelHandle, errorHandle, numThreads);


## model run
### 1. 学习java如何将流储存到数组中
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++]; 
                // byteBuffer Input
                imgData.putFloat(((val >> 16) & 0xFF));
                imgData.putFloat(((val >> 8) & 0xFF));
                imgData.putFloat(val & 0xFF);

### 2. 在c++中将数据流存储为数组形式
### 3. 使用vector存储数组的各项信息