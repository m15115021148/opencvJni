

LOCAL_PATH:=$(call my-dir)

#opencv start
include $(CLEAR_VARS)
OpenCV_INSTALL_MODULES := on
OpenCV_CAMERA_MODULES := off
OPENCV_LIB_TYPE :=STATIC

ifeq ("$(wildcard $(OPENCV_MK_PATH))","")
include $(LOCAL_PATH)/native/jni/OpenCV.mk
else
include $(OPENCV_MK_PATH)
endif
#opecv end

LOCAL_MODULE := OpenCV
LOCAL_SRC_FILES :=  main.cpp

LOCAL_LDLIBS +=  -lm -llog

LOCAL_C_INCLUDES := $(LOCAL_PATH)/native/jni/include

include $(BUILD_SHARED_LIBRARY)

