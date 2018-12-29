
/*
 * JNI helper functions.
 *
 * This file may be included by C or C++ code, which is trouble because jni.h
 * uses different typedefs for JNIEnv in each language.
 *
 * TODO: remove C support.
 */
#ifndef NATIVEHELPER_JNIHELP_H_
#define NATIVEHELPER_JNIHELP_H_

#include "jni.h"
#include <errno.h>
#include <unistd.h>

#ifndef NELEM
# define NELEM(x) ((int) (sizeof(x) / sizeof((x)[0])))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Register one or more native methods with a particular class.
 * "className" looks like "java/lang/String". Aborts on failure.
 * TODO: fix all callers and change the return type to void.
 */
int jniRegisterNativeMethods(JNIEnv* env, const char* className, const JNINativeMethod* gMethods, int numMethods);

#ifdef __cplusplus
}
#endif


#endif 


