package com.example.mnnllmdemo

import android.app.Application
import com.example.mnnllmdemo.llm.DetResultReporter
import com.example.mnnllmdemo.llm.LlmSimpleClient

class MyApp : Application() {
    override fun onCreate() {
        super.onCreate()
        // 提前 init，避免第一次触发才初始化导致卡顿
        DetResultReporter.init(this)
        LlmSimpleClient.init(this)
    }
}
