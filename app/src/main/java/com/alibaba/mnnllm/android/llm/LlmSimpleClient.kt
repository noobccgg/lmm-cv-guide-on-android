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
 * 全局：1个 LlmSession + 流式 TTS。
 *
 * - init(context)：一次性初始化（拷贝模型、加载会话、初始化TTS）
 * - setSystemPrompt(p)：更新系统 Prompt（影响后续 ask）
 * - ask(question, systemPromptOverride?, speak=true)：提问，可选覆盖 system；支持流式边播报
 * - release()：释放
 */
object LlmSimpleClient {
    private const val TAG = "LLM"
    private const val MODEL_DIR_IN_ASSETS = "models/Qwen3-0.6B-MNN"
    private const val ASSISTANT_PROMPT = "<|im_start|>assistant\n%s<|im_end|>\n"
    private const val DEFAULT_SYSTEM = "You are a helpful assistant."

    @Volatile private var session: LlmSession? = null
    private val inited = AtomicBoolean(false)

    // 可动态修改的 system prompt
    @Volatile private var systemPrompt: String = DEFAULT_SYSTEM

    // 语音
    private var ttsSpeaker: TtsSpeaker? = null
    private val ttsCollector = StreamingTtsCollector()

    /** 只初始化一次（线程安全，失败会回滚 inited 标志） */
    fun init(context: Context) {
        if (!inited.compareAndSet(false, true)) return
        try {
            val appCtx = context.applicationContext

            // 1) 拷贝模型目录
            val localModelDir = File(appCtx.filesDir, MODEL_DIR_IN_ASSETS)
            AssetUtils.ensureAssetDirCopied(appCtx, MODEL_DIR_IN_ASSETS, localModelDir)
            Log.d(TAG, "Assets copied to: ${localModelDir.absolutePath}")

            // 2) 初始化 TTS（幂等）
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
                updateAssistantPrompt(ASSISTANT_PROMPT) // 只影响 assistant 格式，不是 system
            }
            Log.d(TAG, "LLM session loaded.")
        } catch (t: Throwable) {
            inited.set(false) // 失败回滚
            throw t
        }
    }

    /** 动态更新 system prompt（生效于后续 ask） */
    fun setSystemPrompt(p: String) {
        systemPrompt = p
    }

    /**
     * 进行一次问答。
     * @param question 用户输入
     * @param systemPromptOverride 临时覆盖 system（可空）
     * @param speak 是否边流式生成边播报
     */
    fun ask(question: String, systemPromptOverride: String? = null, speak: Boolean = true) {
        val s = session
        if (s == null) {
            Log.e(TAG, "ask() called before init().")
            return
        }
        val sys = (systemPromptOverride ?: systemPrompt).ifBlank { DEFAULT_SYSTEM }

        // 便于排查：打印这次真正生效的 system & user
        val pretty = buildString {
            appendLine("下面是一次对话：")
            appendLine("【system】$sys")
            appendLine("【user】$question")
        }
        Log.d(TAG, "---- LLM question start ----\n$pretty---- LLM question end ----")

        // 过滤<think> + 分句回调
        val listener = object : SafeProgressListener() {
            override fun onProgressNullable(token: String?): Boolean {
                if (token.isNullOrEmpty()) return false
                val clean = sanitizeThink(token)
                if (clean.isEmpty()) return false

                Log.d(TAG, clean)

                if (speak) {
                    ttsCollector.push(clean) { sentence ->
                        if (sentence.isNotBlank()) ttsSpeaker?.speak(sentence)
                    }
                }
                return false // 返回 false 继续生成
            }

            override fun onComplete() {
                if (speak) {
                    ttsCollector.flush { sentence ->
                        if (sentence.isNotBlank()) ttsSpeaker?.speak(sentence)
                    }
                }
                Log.d(TAG, "[LLM Complete]")
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
            // 注意：这是阻塞调用；建议在后台线程调用（你已在 DetResultReporter 里后台执行）
            s.generateFromMessages(messages, listener)
        } catch (e: Throwable) {
            Log.e(TAG, "LLM exception: ${e.message}", e)
        }
    }

    /** 可在退出时调用 */
    fun release() {
        try { session?.release() } catch (_: Throwable) {}
        session = null
        inited.set(false)
        ttsSpeaker?.release()
        ttsSpeaker = null
        Log.d(TAG, "LLM session released.")
    }

    // ============== 过滤<think>并按句号/换行切片给TTS ==============

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

    /** 简易过滤（对外复用） */
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

    /** 串行 TTS 播放器 + 音频焦点（只启动一次 worker 线程） */
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
                        // exit
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
