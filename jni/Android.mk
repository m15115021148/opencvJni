
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


include $(CLEAR_VARS)
LOCAL_MODULE := libavcodec
LOCAL_SRC_FILES := ffmpeg/lib/libavcodec.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libavdevice
LOCAL_SRC_FILES := ffmpeg/lib/libavdevice.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libavfilter
LOCAL_SRC_FILES := ffmpeg/lib/libavfilter.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libavformat
LOCAL_SRC_FILES := ffmpeg/lib/libavformat.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libavresample
LOCAL_SRC_FILES := ffmpeg/lib/libavresample.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libavutil
LOCAL_SRC_FILES := ffmpeg/lib/libavutil.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libpostproc
LOCAL_SRC_FILES := ffmpeg/lib/libpostproc.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libswresample
LOCAL_SRC_FILES := ffmpeg/lib/libswresample.a
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := libswscale
LOCAL_SRC_FILES := ffmpeg/lib/libswscale.a
include $(PREBUILT_STATIC_LIBRARY)




LOCAL_MODULE := OpenCV
LOCAL_SRC_FILES := coreApp.cpp


#设置可以使用C++代码  
LOCAL_CPPFLAGS += -std=c++11

LOCAL_STATIC_LIBRARIES := libavcodec \
							libavdevice \
							libavfilter \
							libavformat \
							libavresample \
							libavutil \
							libpostproc \
							libswresample \
							libswscale

LOCAL_C_INCLUDES := $(LOCAL_PATH)/native/jni/include \
					$(LOCAL_PATH)/ffmpeg/include

LOCAL_LDLIBS := -llog -lz -ldl

include $(BUILD_SHARED_LIBRARY)



