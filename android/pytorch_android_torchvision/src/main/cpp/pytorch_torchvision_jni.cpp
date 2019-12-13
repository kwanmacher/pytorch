#include <cassert>
#include <cmath>
#include <vector>

#include <libyuv.h>

#include <fbjni/ByteBuffer.h>
#include <fbjni/fbjni.h>

#if defined(__ANDROID__)

#include <android/log.h>
#define ALOGI(...) \
  __android_log_print(ANDROID_LOG_INFO, "pytorch-vision-jni", __VA_ARGS__)

#endif

namespace pytorch_vision_jni {
class PytorchVisionJni : public facebook::jni::JavaClass<PytorchVisionJni> {
 private:
  static inline int f0X(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return cropXAfterRtn;
  }

  static inline int f0Y(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return cropYAfterRtn;
  }

  static inline int f90X(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return cropYAfterRtn;
  }
  static inline int f90Y(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return (cropHeightBeforeRtn - 1) - cropXAfterRtn;
  }

  static inline int f180X(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return (cropWidthBeforeRtn - 1) - cropXAfterRtn;
  }
  static inline int f180Y(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return (cropHeightBeforeRtn - 1) - cropYAfterRtn;
  }

  static inline int f270X(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return (cropWidthBeforeRtn - 1) - cropYAfterRtn;
  }
  static inline int f270Y(
      int cropWidthBeforeRtn,
      int cropHeightBeforeRtn,
      int cropXAfterRtn,
      int cropYAfterRtn) {
    return cropXAfterRtn;
  }

  typedef int (*const fp)(int, int, int, int);

 public:
  constexpr static auto kJavaDescriptor =
      "Lorg/pytorch/torchvision/PyTorchVision;";

  static inline int clamp(float n, float lower, float upper) {
    return std::max(lower, std::min(n, upper));
  }

  static void nativeImageYUV420CenterCropToFloatBuffer1(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    static JNIEnv* jni = facebook::jni::Environment::current();
    const auto dataCapacity = jni->GetDirectBufferCapacity(outBuffer.get());
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    const auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    const auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    const int widthBeforeRtn = imageWidth;
    const int heightBeforeRtn = imageHeight;

    int widthAfterRtn = widthBeforeRtn;
    int heightAfterRtn = heightBeforeRtn;
    bool oddRotation = rotateCWDegrees == 90 || rotateCWDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = heightBeforeRtn;
      heightAfterRtn = widthBeforeRtn;
    }

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;

    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn =
          std::floor(tensorWidth * heightAfterRtn / tensorHeight);
    } else {
      cropHeightAfterRtn =
          std::floor(tensorHeight * widthAfterRtn / tensorWidth);
    }

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      std::swap(cropWidthBeforeRtn, cropHeightBeforeRtn);
    }

    const int offsetX = std::floor((widthBeforeRtn - cropWidthBeforeRtn) / 2.f);
    const int offsetY =
        std::floor((heightBeforeRtn - cropHeightBeforeRtn) / 2.f);

    const uint8_t* yData = yBuffer->getDirectBytes();
    const uint8_t* uData = uBuffer->getDirectBytes();
    const uint8_t* vData = vBuffer->getDirectBytes();

    float scale = cropWidthAfterRtn / tensorWidth;
    int uvRowStride = uRowStride >> 1;

    int channelSize = tensorHeight * tensorWidth;
    int tensorInputOffsetG = channelSize;
    int tensorInputOffsetB = channelSize << 1;

    static fp pfX[] = {f0X, f90X, f180X, f270X};
    static fp pfY[] = {f0Y, f90Y, f180Y, f270Y};
    int fidx = rotateCWDegrees / 90;
    fp rfX = pfX[fidx];
    fp rfY = pfY[fidx];

    for (int y = 0; y < tensorHeight; y++) {
      for (int x = 0; x < tensorWidth; x++) {
        int cropXAfterRtn = std::floor(x * scale);
        int cropYAfterRtn = std::floor(y * scale);

        int xBeforeRtn =
            rfX(cropWidthBeforeRtn,
               cropHeightBeforeRtn,
               cropXAfterRtn,
               cropYAfterRtn);
        int yBeforeRtn =
            rfY(cropWidthBeforeRtn,
               cropHeightBeforeRtn,
               cropXAfterRtn,
               cropYAfterRtn);

        int yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        int uvIdx =
            (yBeforeRtn >> 1) * uvRowStride + xBeforeRtn * uvPixelStride;

        const int yi = yData[yIdx];
        const int ui = uData[uvIdx];
        const int vi = vData[uvIdx];

        const int a0 = 1192 * (yi - 16);
        const int a1 = 1634 * (vi - 128);
        const int a2 = 832 * (vi - 128);
        const int a3 = 400 * (ui - 128);
        const int a4 = 2066 * (ui - 128);

        const int r = clamp((a0 + a1) >> 10, 0, 255);
        const int g = clamp((a0 - a2 - a3) >> 10, 0, 255);
        const int b = clamp((a0 + a4) >> 10, 0, 255);

        const float rf = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        const float gf = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        const float bf = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        const int offset = outOffset + y * tensorWidth + x;
        outData[offset] = rf;
        outData[offset + tensorInputOffsetG] = gf;
        outData[offset + tensorInputOffsetB] = bf;
      }
    }
  }

  static void nativeImageYUV420CenterCropToFloatBuffer2(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    static JNIEnv* jni = facebook::jni::Environment::current();
    const auto dataCapacity = jni->GetDirectBufferCapacity(outBuffer.get());
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    const auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    const auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    const int widthBeforeRtn = imageWidth;
    const int heightBeforeRtn = imageHeight;

    int widthAfterRtn = widthBeforeRtn;
    int heightAfterRtn = heightBeforeRtn;
    bool oddRotation = rotateCWDegrees == 90 || rotateCWDegrees == 270;
    if (oddRotation) {
      widthAfterRtn = heightBeforeRtn;
      heightAfterRtn = widthBeforeRtn;
    }

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;

    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn =
          std::floor(tensorWidth * heightAfterRtn / tensorHeight);
    } else {
      cropHeightAfterRtn =
          std::floor(tensorHeight * widthAfterRtn / tensorWidth);
    }

    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (oddRotation) {
      std::swap(cropWidthBeforeRtn, cropHeightBeforeRtn);
    }

    const int offsetX = std::floor((widthBeforeRtn - cropWidthBeforeRtn) / 2.f);
    const int offsetY =
        std::floor((heightBeforeRtn - cropHeightBeforeRtn) / 2.f);

    const uint8_t* yData = yBuffer->getDirectBytes();
    const uint8_t* uData = uBuffer->getDirectBytes();
    const uint8_t* vData = vBuffer->getDirectBytes();

    float scale = cropWidthAfterRtn / tensorWidth;
    int uvRowStride = uRowStride >> 1;

    int channelSize = tensorHeight * tensorWidth;
    int tensorInputOffsetG = channelSize;
    int tensorInputOffsetB = channelSize << 1;

    for (int y = 0; y < tensorHeight; y++) {
      for (int x = 0; x < tensorWidth; x++) {
        int cropXAfterRtn = std::floor(x * scale);
        int cropYAfterRtn = std::floor(y * scale);

        int xBeforeRtn = offsetX + cropXAfterRtn;
        int yBeforeRtn = offsetY + cropYAfterRtn;

        if (rotateCWDegrees == 90) {
          xBeforeRtn = offsetX + cropYAfterRtn;
          yBeforeRtn = offsetY + (cropHeightBeforeRtn - 1) - cropXAfterRtn;
        } else if (rotateCWDegrees == 180) {
          xBeforeRtn = offsetX + (cropWidthBeforeRtn - 1) - cropXAfterRtn;
          yBeforeRtn = offsetY + (cropHeightBeforeRtn - 1) - cropYAfterRtn;
        } else if (rotateCWDegrees == 270) {
          xBeforeRtn = offsetX + (cropWidthBeforeRtn - 1) - cropYAfterRtn;
          yBeforeRtn = offsetY + cropXAfterRtn;
        }

        int yIdx = yBeforeRtn * yRowStride + xBeforeRtn * yPixelStride;
        int uvIdx =
            (yBeforeRtn >> 1) * uvRowStride + xBeforeRtn * uvPixelStride;

        const int yi = yData[yIdx];
        const int ui = uData[uvIdx];
        const int vi = vData[uvIdx];

        const int a0 = 1192 * (yi - 16);
        const int a1 = 1634 * (vi - 128);
        const int a2 = 832 * (vi - 128);
        const int a3 = 400 * (ui - 128);
        const int a4 = 2066 * (ui - 128);

        const int r = clamp((a0 + a1) >> 10, 0, 255);
        const int g = clamp((a0 - a2 - a3) >> 10, 0, 255);
        const int b = clamp((a0 + a4) >> 10, 0, 255);

        const float rf = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        const float gf = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        const float bf = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        const int offset = outOffset + y * tensorWidth + x;
        outData[offset] = rf;
        outData[offset + tensorInputOffsetG] = gf;
        outData[offset + tensorInputOffsetB] = bf;
      }
    }
  }

  static void nativeImageYUV420CenterCropToFloatBufferLibyuv(
      facebook::jni::alias_ref<jclass>,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> yBuffer,
      const int yRowStride,
      const int yPixelStride,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> uBuffer,
      facebook::jni::alias_ref<facebook::jni::JByteBuffer> vBuffer,
      const int uRowStride,
      const int uvPixelStride,
      const int imageWidth,
      const int imageHeight,
      const int rotateCWDegrees,
      const int tensorWidth,
      const int tensorHeight,
      facebook::jni::alias_ref<jfloatArray> jnormMeanRGB,
      facebook::jni::alias_ref<jfloatArray> jnormStdRGB,
      facebook::jni::alias_ref<facebook::jni::JBuffer> outBuffer,
      const int outOffset) {
    const int halfImageWidth = (imageWidth + 1) / 2;
    const int halfImageHeight = (imageHeight + 1) / 2;

    // widthBeforeRtn, heightBeforeRtn{
    const int widthBeforeRtn = imageWidth;
    const int heightBeforeRtn = imageHeight;
    int widthAfterRtn = widthBeforeRtn;
    int heightAfterRtn = heightBeforeRtn;
    // rotatedStrideU, rotatedStrideV{
    int rotatedStrideU = halfImageWidth;
    int rotatedStrideV = halfImageHeight;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      widthAfterRtn = heightBeforeRtn;
      heightAfterRtn = widthBeforeRtn;
      rotatedStrideU = halfImageHeight;
      rotatedStrideV = halfImageWidth;
    }
    // }widthBeforeRtn, heightBeforeRtn
    // }rotatedStrideU, rotatedStrideV

    int cropWidthAfterRtn = widthAfterRtn;
    int cropHeightAfterRtn = heightAfterRtn;
    if (tensorWidth * heightAfterRtn <= tensorHeight * widthAfterRtn) {
      cropWidthAfterRtn =
          std::floor(tensorWidth * heightAfterRtn / tensorHeight);
    } else {
      cropHeightAfterRtn =
          std::floor(tensorHeight * widthAfterRtn / tensorWidth);
    }
    // }cropWidthAfterRtn, cropHeightAfterRtn
    const int halfCropWidthAfterRtn = (cropWidthAfterRtn + 1) / 2;
    const int halfCropHeightAfterRtn = (cropHeightAfterRtn + 1) / 2;

    const int cropXAfterRtn = (widthAfterRtn - cropWidthAfterRtn) / 2;
    const int cropYAfterRtn = (heightAfterRtn - cropHeightAfterRtn) / 2;

    int cropXBeforeRtn = cropXAfterRtn;
    int cropYBeforeRtn = cropYAfterRtn;
    int cropWidthBeforeRtn = cropWidthAfterRtn;
    int cropHeightBeforeRtn = cropHeightAfterRtn;
    if (rotateCWDegrees == 90 || rotateCWDegrees == 270) {
      std::swap(cropXBeforeRtn, cropYBeforeRtn);
      std::swap(cropWidthBeforeRtn, cropHeightBeforeRtn);
    }
    // }cropXBeforeRtn, cropYBeforeRtn
    // }cropWidthBeforeRtn, cropHeightBeforeRtn
    const int halfCropWidthBeforeRtn = (cropWidthBeforeRtn + 1) / 2;
    const int halfCropHeightBeforeRtn = (cropHeightBeforeRtn + 1) / 2;

    const uint32_t i420CropSize = cropWidthAfterRtn * cropHeightAfterRtn;
    std::vector<uint8_t> i420Crop;
    if (i420Crop.size() != i420CropSize) {
      i420Crop.resize(i420CropSize);
    }

    uint8_t* i420CropY = i420Crop.data();
    uint8_t* i420CropU = i420CropY + cropWidthBeforeRtn * cropWidthBeforeRtn;
    uint8_t* i420CropV =
        i420CropU + halfCropWidthBeforeRtn * halfCropHeightBeforeRtn;

    const auto retAndroid420ToI420 = libyuv::Android420ToI420(
        yBuffer->getDirectBytes() + cropYBeforeRtn * yRowStride +
            cropXBeforeRtn,
        yRowStride,
        uBuffer->getDirectBytes() + cropYBeforeRtn * uRowStride +
            cropXBeforeRtn * uvPixelStride,
        uRowStride,
        vBuffer->getDirectBytes() + cropYBeforeRtn * uRowStride +
            cropXBeforeRtn * uvPixelStride,
        uRowStride,
        uvPixelStride,
        i420CropY,
        imageWidth,
        i420CropU,
        halfImageWidth,
        i420CropV,
        halfImageHeight,
        cropWidthBeforeRtn,
        cropHeightBeforeRtn);
    assert(retAndroid420ToI420 == 0);

    // Rotate{
    const uint32_t i420CropRtdSize = cropWidthAfterRtn * cropHeightAfterRtn;
    std::vector<uint8_t> i420CropRtd;
    if (i420CropRtd.size() != i420CropRtdSize) {
      i420CropRtd.resize(i420CropRtdSize);
    }

    uint8_t* i420CropRtdY = i420CropRtd.data();
    uint8_t* i420CropRtdU =
        i420CropRtdY + cropWidthAfterRtn * cropWidthAfterRtn;
    uint8_t* i420CropRtdV =
        i420CropRtdU + halfCropWidthAfterRtn * halfCropWidthAfterRtn;
    libyuv::RotationMode rmode = libyuv::RotationMode::kRotate0;
    if (rotateCWDegrees == 90) {
      rmode = libyuv::RotationMode::kRotate90;
    } else if (rotateCWDegrees == 180) {
      rmode = libyuv::RotationMode::kRotate180;
    } else if (rotateCWDegrees == 270) {
      rmode = libyuv::RotationMode::kRotate270;
    }

    const auto retI420Rotate = libyuv::I420Rotate(
        i420CropY,
        cropWidthBeforeRtn,
        i420CropU,
        halfCropWidthBeforeRtn,
        i420CropV,
        halfCropHeightBeforeRtn,
        i420CropRtdY,
        cropWidthAfterRtn,
        i420CropRtdU,
        halfCropWidthAfterRtn,
        i420CropRtdV,
        halfCropHeightAfterRtn,
        cropWidthAfterRtn,
        cropHeightAfterRtn,
        rmode);
    assert(retI420Rotate == 0);
    // }Rotate

    // ARGBScale{
    const uint32_t argbTensorSize = 4 * tensorWidth * tensorHeight;
    std::vector<uint8_t> argbTensor;
    if (argbTensor.size() != argbTensorSize) {
      argbTensor.resize(argbTensorSize);
    }

    uint8_t* argbData = argbTensor.data();
    const auto retYUVToARGBScaleClip = libyuv::YUVToARGBScaleClip(
        i420CropRtdY,
        cropWidthAfterRtn,
        i420CropRtdU,
        halfCropWidthAfterRtn,
        i420CropRtdV,
        halfCropHeightAfterRtn,
        libyuv::FOURCC_I420,
        cropWidthAfterRtn,
        cropWidthAfterRtn,
        argbData,
        4 * tensorWidth,
        libyuv::FOURCC_ARGB,
        tensorWidth,
        tensorHeight,
        0,
        0,
        tensorWidth,
        tensorHeight,
        libyuv::FilterMode::kFilterNone);
    assert(retYUVToARGBScaleClip == 0);
    // }ARGBScale

    JNIEnv* jni = facebook::jni::Environment::current();
    const auto dataCapacity = jni->GetDirectBufferCapacity(outBuffer.get());
    float* outData = (float*)jni->GetDirectBufferAddress(outBuffer.get());

    int channelSize = tensorHeight * tensorWidth;
    int tensorInputOffsetG = channelSize;
    int tensorInputOffsetB = 2 * channelSize;
    const auto normMeanRGB = jnormMeanRGB->getRegion(0, 3);
    const auto normStdRGB = jnormStdRGB->getRegion(0, 3);

    for (int x = 0; x < tensorWidth; x++) {
      for (int y = 0; y < tensorHeight; y++) {
        int offset = y * tensorWidth + x;
        const int r = argbData[channelSize + offset];
        const int g = argbData[2 * channelSize + offset];
        const int b = argbData[3 * channelSize + offset];

        const float rf = ((r / 255.f) - normMeanRGB[0]) / normStdRGB[0];
        const float gf = ((g / 255.f) - normMeanRGB[1]) / normStdRGB[1];
        const float bf = ((b / 255.f) - normMeanRGB[2]) / normStdRGB[2];

        outData[outOffset + offset] = rf;
        outData[outOffset + offset + tensorInputOffsetG] = gf;
        outData[outOffset + offset + tensorInputOffsetB] = bf;
      }
    }
  }

  static void registerNatives() {
    javaClassStatic()->registerNatives({
        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBuffer1",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBuffer1),

        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBuffer2",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBuffer2),

        makeNativeMethod(
            "nativeImageYUV420CenterCropToFloatBufferLibyuv",
            PytorchVisionJni::nativeImageYUV420CenterCropToFloatBufferLibyuv),
    });
  }
};
} // namespace pytorch_vision_jni

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(
      vm, [] { pytorch_vision_jni::PytorchVisionJni::registerNatives(); });
}