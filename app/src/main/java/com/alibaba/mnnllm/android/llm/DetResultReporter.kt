package com.example.mnnllmdemo.llm

import android.content.Context
import android.os.Process
import android.util.Log
import java.util.Locale
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * 逐条/批量收集识别结果，满50条自动发给 LLM（或手动 flush）。
 * - offer(label, score, depth)        ：逐条喂入
 * - offerRaw(line)                    ：喂入一条自定义文本（已拼好的行）
 * - flushNow(context)                 ：不足50也强制发送
 *
 * 设计要点：
 * 1) LLM 调用放到单线程后台池，降低对相机/GL 的干扰；
 * 2) 同时只允许 1 个在跑；忙时仅保留“最后一批”pending，避免积压；
 * 3) 日志只打摘要，避免超长 log 卡 UI；
 * 4) 对 “一行/两行一物体” 都兼容（可选的合并逻辑见 MERGE 标记处）。
 */
object DetResultReporter {
    private const val TAG = "LLM"
    private const val BATCH_SIZE = 50

    @Volatile private var appCtx: Context? = null

    // 缓存与并发控制
    private val lock = Any()
    private val buffer = mutableListOf<String>()

    // LLM 后台执行器（单线程 + 后台优先级）
    private val executor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "llm-worker").apply {
            priority = Thread.NORM_PRIORITY - 2
        }
    }
    private val llmBusy = AtomicBoolean(false)
    @Volatile private var pendingBatch: List<String>? = null

    /** 建议在 Application.onCreate() 里调用一次 */
    @JvmStatic
    fun init(context: Context) {
        appCtx = context.applicationContext
    }

    /** 把结构化结果格式化为 "label,0.89,2.35" */
    @JvmStatic
    fun offer(label: String, score: Float, depthMeters: Float) {
        val line = buildString {
            append(label)
            append(',')
            append(String.format(Locale.US, "%.2f", score))
            append(',')
            append(String.format(Locale.US, "%.2f", depthMeters))
        }
        offerRaw(line)
    }

    /** 喂入一行文本。行内如果已包含 bbox（例如 "xxx | Left: ..."），直接入缓存。 */
    @JvmStatic
    fun offerRaw(line: String) {
        val trimmed = line.trim()
        if (trimmed.isEmpty()) return

        // ===== 可选：MERGE 两行一目标的容错示例 =====
        // 如果你的 GLRender 改成“标题行 + bbox行”分两次喂，这里可以把上一行缓存起来，遇到以 "Left:" 开头的行就合并为一行再入队。
        // 当前你已改为“一行内含 | Left: ...”，所以可以不需要这段；保留注释供以后切换。
        // ======================================

        var ready: List<String>? = null
        synchronized(lock) {
            buffer.add(trimmed)
            if (buffer.size >= BATCH_SIZE) {
                ready = buffer.take(BATCH_SIZE).toList()
                buffer.clear()
            }
        }
        if (ready != null) sendBatch(ready!!)
    }

    /** 强制把当前缓存发给 LLM（不足 BATCH_SIZE 也发），发完清空缓存 */
    @JvmStatic
    fun flushNow(context: Context? = null) {
        if (context != null) appCtx = context.applicationContext
        val snapshot: List<String>
        synchronized(lock) {
            if (buffer.isEmpty()) {
                Log.d(TAG, "flushNow: empty, skip.")
                return
            }
            snapshot = buffer.toList()
            buffer.clear()
        }
        sendBatch(snapshot)
    }

    // ================== 内部实现 ==================

    private fun sendBatch(lines: List<String>) {
        val ctx = appCtx
        if (ctx == null) {
            Log.w(TAG, "sendBatchToLlm: appCtx is null, did you call DetResultReporter.init(context)?")
            // 仍然继续，LlmSimpleClient.init 会在有 ctx 时生效
        } else {
            // 幂等
            LlmSimpleClient.init(ctx)
        }

        // 忙：只保留“最新一批”；不堆积多批
        if (!llmBusy.compareAndSet(false, true)) {
            pendingBatch = lines // 覆盖上一批待发
            Log.w(TAG, "LLM busy, replace pending batch (size=${lines.size})")
            return
        }

        // 真正发送放后台
        val prompt = buildPrompt(lines)
        executor.execute {
            // 把线程优先级降到 BACKGROUND，进一步减少对前台的影响
            try {
                Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND)
            } catch (_: Throwable) {}

            // 控制日志量：只打首尾行数
            Log.d(TAG, "Trigger LLM batch (objects) size=${lines.size}")

            try {
                LlmSimpleClient.ask(prompt)  // 假定是同步阻塞的方法
            } catch (t: Throwable) {
                Log.e(TAG, "LLM call failed: ${t.message}", t)
            } finally {
                // 释放 busy 并检查是否有挂起批次
                llmBusy.set(false)
                val toSend = pendingBatch
                if (toSend != null) {
                    pendingBatch = null
                    sendBatch(toSend) // 递归触发下一批
                }
            }
        }
    }

    private fun buildPrompt(top: List<String>): String {
        return buildString {
            appendLine("下面是最近采集的最多 $BATCH_SIZE 个目标，请用简洁中文给出概括：")
            appendLine("1) 主要出现了哪些类别？")
            appendLine("2) 有无可疑或安全风险信息？")
            appendLine("3) 用一句话总结现场情况。")
            appendLine()
            appendLine("【识别结果】")
            top.forEachIndexed { i, s -> append(i + 1).append(". ").appendLine(s) }
        }
    }
}
