LOCAL_PATH:=$(call my-dir)

#opencv start

include $(CLEAR_VARS)

OPENCVROOT :=/Users/cmm/Desktop/work/opencv-ffmpeg/android_opencv/opencv
OpenCV_INSTALL_MODULES := on
OpenCV_CAMERA_MODULES := off
OPENCV_LIB_TYPE :=STATIC

include ${OPENCVROOT}/sdk/native/jni/OpenCV.mk

#opecv end

LOCAL_MODULE := OpenCV
LOCAL_SRC_FILES := coreApp.cpp

LOCAL_FORCE_STATIC_EXECUTABLE := true

#设置可以使用C++代码  
LOCAL_CPPFLAGS += -std=c++11

LOCAL_C_INCLUDES := $(LOCAL_PATH)/native/jni/include

LOCAL_LDLIBS += -lavformat -lavcodec -lx264 -lswresample -lswscale -lavutil -lm -llog -landroid

include $(BUILD_SHARED_LIBRARY)

