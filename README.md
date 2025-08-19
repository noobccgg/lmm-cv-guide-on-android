assets、models里放tts和语言模型。
我把GLrender里面渲染照相机预览那个关了，然后把目标识别的频率也降低了，想着让推理变快点。
新增的文件是DetResultReporter和LlmSimpleClient
mainactivity是在下面这里调用的DetResultReporter
    depth_view.setText("Central\nDepth: " + String.format("%.2f", central_depth) + " m");
                        class_view.setText(class_result);
                        String snapshot = class_result.toString();
                        class_view.setText(snapshot);

                        // === 新增：把这一帧的每一行喂给 DDetResultReporter ===
                        if (!snapshot.isEmpty()) {
                            String[] lines = snapshot.split("\\n");
                            for (String line : lines) {
                                line = line.trim();
                                if (line.isEmpty()) continue;
                                // 你的格式是 "label / 95.3% / 1.2 m"
                                try {
                                    String[] segs = line.split("/");
                                    String label = segs[0].trim();

                                    float score = 0f;
                                    if (segs.length > 1) {
                                        String s = segs[1].replace("%","").trim();
                                        score = Float.parseFloat(s) / 100f;
                                    }

                                    float depthMeters = 0f;
                                    if (segs.length > 2) {
                                        String d = segs[2].replace("m","").trim();
                                        depthMeters = Float.parseFloat(d);
                                    }

                                    // 把“解析后的单条结果”喂给 Reporter
                                    DetResultReporter.INSTANCE.offer(label, score, depthMeters);

                                } catch (Throwable ignore) {
                                    // 单行解析失败就忽略，避免影响整帧
                                }

    然后转到DetResultReporter处理，处理完以后这个文件最后面开了一个线程专门给大模型，具体细节我还没研究。现在输入信息很简单就是一个攒够五十行就输给大模型。

    现在老是崩貌似是因为运行流程没弄好，就比如说tts上一句还没播完又来了下一句，然后大语言模型还没输出玩又来了下一个问题这样。

    还有一个问题就是不知道大语言模型调用哪里那个systemprompt有没有用起来。

    
