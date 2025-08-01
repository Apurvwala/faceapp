[app]
title = FaceApp
package.name = faceapp
package.domain = org.yourdomain
source.dir = .
source.include_exts = py,kv,png,jpg,json,atlas

version = 0.1

requirements = python3,kivy,opencv-python,requests,pandas,fpdf,plyer,edge-tts,pygame,apscheduler

android.permissions = CAMERA, RECORD_AUDIO, WRITE_EXTERNAL_STORAGE, READ_EXTERNAL_STORAGE
android.archs = arm64-v8a, armeabi-v7a
android.minapi = 21
android.api = 31
android.ndk = 28c
android.ndk_api = 21
android.private_storage = False

orientation = portrait

[buildozer]
log_level = 2
warn_on_root = 1
