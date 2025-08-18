package com.example.mnnllmdemo.llm

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.util.Log
import com.alibaba.mnnllm.android.llm.LlmSession
import com.benjaminwan.chinesettsmodule.TtsEngine
import com.example.mnnllmdemo.SafeProgressListener
import com.example.mnnllmdemo.util.AssetUtils
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.concurrent.thread

/**
 * 轻量封装：全局驻留一个 LlmSession + 流式TTS。
 * - init(context)：一次性初始化（拷贝模型、加载会话、初始化TTS）
 * - ask(question, systemPrompt?, speak)：带系统prompt地发问，并可实时播报
 * - setSystemPrompt(p)：动态修改系统prompt
 * - release()：释放
 */
object LlmSimpleClient {
    private const val TAG = "LLM"
    private const val MODEL_DIR_IN_ASSETS = "models/Qwen3-0.6B-MNN"
    private const val ASSISTANT_PROMPT = "<|im_start|>assistant\n%s<|im_end|>\n"
    private const val DEFAULT_SYSTEM = "You are a helpful assistant."

    @Volatile private var session: LlmSession? = null
    private val inited = AtomicBoolean(false)

    // System prompt 可动态修改
    @Volatile private var systemPrompt: String = DEFAULT_SYSTEM

    // 语音相关
    private var ttsSpeaker: TtsSpeaker? = null
    private val ttsCollector = StreamingTtsCollector()

    /** 只初始化一次；建议在 Application.onCreate 调用 */
    fun init(context: Context) {
        if (inited.get()) return
        synchronized(this) {
            if (inited.get()) return

            val appCtx = context.applicationContext

            // 1) 拷贝 assets → 内部目录
            val localModelDir = File(appCtx.filesDir, MODEL_DIR_IN_ASSETS)
            AssetUtils.ensureAssetDirCopied(appCtx, MODEL_DIR_IN_ASSETS, localModelDir)
            Log.d(TAG, "Assets copied to: ${localModelDir.absolutePath}")

            // 2) 初始化 TTS（一次即可）
            try {
                TtsEngine.init(appCtx)
                ttsSpeaker = TtsSpeaker(appCtx)
                Log.d(TAG, "TTS initialized.")
            } catch (e: Throwable) {
                Log.w(TAG, "Init TTS failed: ${e.message}", e)
            }

            // 3) 创建并加载 LLM 会话
            val configPath = File(localModelDir, "config.json").absolutePath
            session = LlmSession(
                modelId = "Qwen3-0.6B-MNN",
                sessionId = System.currentTimeMillis().toString(),
                configPath = configPath,
                savedHistory = null
            ).apply {
                setKeepHistory(false)
                setHistory(null)
                load()
                updateAssistantPrompt(ASSISTANT_PROMPT)
            }
            Log.d(TAG, "LLM session loaded.")

            inited.set(true)
        }
    }

    /** 可随时修改系统 prompt（影响后续 ask） */
    fun setSystemPrompt(p: String) { systemPrompt = p }

    /**
     * 传入用户问题；可指定临时的systemPrompt与是否播报。
     * - speak=true：边流式生成边播报
     */
    fun ask(question: String, systemPromptOverride: String? = null, speak: Boolean = true) {
        val s = session
        if (s == null) {
            Log.e(TAG, "ask() called before init().")
            return
        }

        val sys = (systemPromptOverride ?: systemPrompt).ifBlank { DEFAULT_SYSTEM }

        val pretty = buildString {
            appendLine("下面是一次对话：")
            appendLine("【system】$sys")
            appendLine("【user】$question")
        }
        Log.d(TAG, "---- LLM question start ----\n$pretty\n---- LLM question end ----")

        // 监听器：过滤<think>并实时分句
        val listener = object : SafeProgressListener() {
            override fun onProgressNullable(token: String?): Boolean {
                if (token.isNullOrEmpty()) return false
                val clean = sanitizeThink(token)
                if (clean.isEmpty()) return false

                // 日志看流式片段
                Log.d(TAG, clean)

                // 分句 -> TTS
                if (speak) {
                    ttsCollector.push(clean) { sentence ->
                        if (sentence.isNotBlank()) ttsSpeaker?.speak(sentence)
                    }
                }
                return false // 必须返回 false，继续生成
            }

            override fun onComplete() {
                if (speak) {
                    ttsCollector.flush { sentence ->
                        if (sentence.isNotBlank()) ttsSpeaker?.speak(sentence)
                    }
                }
                Log.d(TAG, "\n[LLM Complete]")
            }

            override fun onError(message: String) {
                Log.e(TAG, "[LLM Error] $message")
            }
        }

        try {
            val messages = listOf(
                "system" to sys,
                "user" to question
            )
            s.generateFromMessages(messages, listener)
        } catch (e: Throwable) {
            Log.e(TAG, "LLM exception: ${e.message}", e)
        }
    }

    /** 退出时可调用一次 */
    fun release() {
        try { session?.release() } catch (_: Throwable) {}
        session = null
        inited.set(false)
        ttsSpeaker?.release()
        ttsSpeaker = null
        Log.d(TAG, "LLM session released.")
    }

    // ============== 辅助：过滤<think>并按句号切片给TTS ==============

    private class StreamingTtsCollector {
        private val sb = StringBuilder()
        private var muteThink = false
        private val boundary = Regex("[。！？!?；;~～…\\n]")

        fun push(raw: String?, onSentence: (String) -> Unit) {
            if (raw.isNullOrEmpty()) return
            var token = sanitizeThinkInternal(raw)
            if (token.isEmpty()) return
            if (sb.isEmpty()) token = token.trimStart()
            sb.append(token)

            var text = sb.toString()
            while (true) {
                val m = boundary.find(text) ?: break
                val end = m.range.last + 1
                val sentence = text.substring(0, end).trim()
                if (sentence.isNotEmpty()) onSentence(sentence)
                text = text.substring(end)
            }
            sb.clear()
            sb.append(text)
        }

        fun flush(onSentence: (String) -> Unit) {
            val tail = sb.toString().trim()
            if (tail.isNotEmpty()) onSentence(tail)
            sb.clear()
        }

        private fun sanitizeThinkInternal(input: String): String {
            var s = input
            if (muteThink) {
                val end = s.indexOf("</think>")
                if (end >= 0) {
                    muteThink = false
                    s = s.substring(end + "</think>".length)
                } else return ""
            }
            while (true) {
                val start = s.indexOf("<think>")
                if (start < 0) break
                val end = s.indexOf("</think>", start)
                s = if (end >= 0) {
                    s.removeRange(start, end + "</think>".length)
                } else {
                    muteThink = true
                    s.substring(0, start)
                }
            }
            return s
        }
    }

    /** 对外也可重用的简易过滤 */
    private fun sanitizeThink(input: String): String {
        var s = input
        while (true) {
            val start = s.indexOf("<think>")
            if (start < 0) break
            val end = s.indexOf("</think>", start)
            s = if (end >= 0) s.removeRange(start, end + "</think>".length) else s.substring(0, start)
        }
        return s
    }

    /** 串行 TTS 播放器 + 音频焦点 */
    private class TtsSpeaker(ctx: Context) {
        private val queue = LinkedBlockingQueue<String>()
        private val workerStarted = AtomicBoolean(false)
        @Volatile private var running = true

        private val audioManager = ctx.getSystemService(Context.AUDIO_SERVICE) as AudioManager
        private val focusRequest: AudioFocusRequest? =
            if (Build.VERSION.SDK_INT >= 26) {
                AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK)
                    .setAudioAttributes(
                        AudioAttributes.Builder()
                            .setUsage(AudioAttributes.USAGE_ASSISTANT)
                            .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                            .build()
                    )
                    .setOnAudioFocusChangeListener { /* no-op */ }
                    .build()
            } else null

        fun speak(sentence: String) {
            if (sentence.isBlank()) return
            queue.offer(sentence)
            ensureWorker()
        }

        private fun ensureWorker() {
            if (workerStarted.compareAndSet(false, true)) {
                thread(name = "TtsSpeaker") {
                    try {
                        // 等 TTS 初始化
                        while (running && !TtsEngine.isInitialized()) Thread.sleep(30)

                        requestFocus()
                        while (running) {
                            val s = queue.take()
                            if (s.isNotBlank()) {
                                try {
                                    // 第二个参数 true：允许内部做轻度分句
                                    TtsEngine.speak(s, true)
                                } catch (_: Throwable) { /* ignore one */ }
                            }
                        }
                    } catch (_: InterruptedException) {
                    } finally {
                        abandonFocus()
                    }
                }
            }
        }

        private fun requestFocus() {
            try {
                if (Build.VERSION.SDK_INT >= 26) {
                    audioManager.requestAudioFocus(focusRequest!!)
                } else {
                    @Suppress("DEPRECATION")
                    audioManager.requestAudioFocus(
                        null,
                        AudioManager.STREAM_MUSIC,
                        AudioManager.AUDIOFOCUS_GAIN_TRANSIENT_MAY_DUCK
                    )
                }
            } catch (_: Throwable) {}
        }

        private fun abandonFocus() {
            try {
                if (Build.VERSION.SDK_INT >= 26) {
                    audioManager.abandonAudioFocusRequest(focusRequest!!)
                } else {
                    @Suppress("DEPRECATION")
                    audioManager.abandonAudioFocus(null)
                }
            } catch (_: Throwable) {}
        }

        fun release() {
            running = false
            abandonFocus()
        }
    }
}
