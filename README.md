# Tflite
c++ implement tflite android demo

## load model 
errorHandle = createErrorReporter(ERROR_BUFFER_SIZE);
modelHandle = createModel(modelPath, errorHandle);
interpreterHandle = createInterpreter(modelHandle, errorHandle, numThreads);


## model run
### 1. resize picture
### 2. 

 Tensor[] run(Object[] inputs) {
    if (inputs == null || inputs.length == 0) {
      throw new IllegalArgumentException("Input error: Inputs should not be null or empty.");
    }
    int[] dataTypes = new int[inputs.length];
    Object[] sizes = new Object[inputs.length];
    int[] numsOfBytes = new int[inputs.length];
    for (int i = 0; i < inputs.length; ++i) {
      DataType dataType = dataTypeOf(inputs[i]);
      Log.d("wwww", "dataType == " +dataType);
      dataTypes[i] = dataType.getNumber();
      Log.d("wwww", "111111111111111");
      if (dataType == DataType.BYTEBUFFER) {
    	  Log.d("wwww", "2222222222222");
        ByteBuffer buffer = (ByteBuffer) inputs[i];
        if (buffer.order() != ByteOrder.nativeOrder()) {
          throw new IllegalArgumentException(
              "Input error: ByteBuffer shoud use ByteOrder.nativeOrder().");
        }
        numsOfBytes[i] = buffer.limit();
        sizes[i] = getInputDims(interpreterHandle, i, numsOfBytes[i]);
      } else if (isNonEmptyArray(inputs[i])) {
    	  Log.d("wwww", "33333333333333");
        int[] dims = shapeOf(inputs[i]);
        sizes[i] = dims;
        numsOfBytes[i] = dataType.elemByteSize() * numElements(dims);
      } else {
    	  Log.d("wwww", "44444444444");
        throw new IllegalArgumentException(
            String.format(
                "Input error: %d-th element of the %d inputs is not an array or a ByteBuffer.",
                i, inputs.length));
      }
    }
    inferenceDurationNanoseconds = -1;
    long[] outputsHandles =
        run(
            interpreterHandle,
            errorHandle,
            sizes,
            dataTypes,
            numsOfBytes,
            inputs,
            this,
            isMemoryAllocated);
    Log.d("OutputsHandles:-------", Integer.toString(outputsHandles.length));
    if (outputsHandles == null || outputsHandles.length == 0) {
      throw new IllegalStateException("Internal error: Interpreter has no outputs.");
    }
    isMemoryAllocated = true;
    Tensor[] outputs = new Tensor[outputsHandles.length];
    for (int i = 0; i < outputsHandles.length; ++i) {
      outputs[i] = Tensor.fromHandle(outputsHandles[i]);
    }
    return outputs;
  }


private static native long[] run(
      long interpreterHandle,
      long errorHandle,
      Object[] sizes,
      int[] dtypes,
      int[] numsOfBytes,
      Object[] values,
      NativeInterpreterWrapper wrapper,
      boolean memoryAllocated);