package com.nageoffer.ai.tinyrag.sse;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.TimeUnit;

public class SseStreamClient {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "sk-xxx"; // 替换为你的 API Key

    // ========== 回调接口 ==========

    /**
     * SSE 流式响应的事件回调
     */
    interface StreamCallback {
        /**
         * 收到一个 content 增量（每个 token 调用一次）
         */
        void onToken(String token);

        /**
         * 流正常结束，返回完整内容和 Token 统计
         */
        void onComplete(String fullContent, Usage usage);

        /**
         * 发生错误，partialContent 是错误发生前已接收到的内容
         */
        void onError(Exception e, String partialContent);
    }

    /**
     * Token 用量统计
     */
    static class Usage {
        int promptTokens;
        int completionTokens;
        int totalTokens;

        @Override
        public String toString() {
            return String.format("prompt=%d, completion=%d, total=%d",
                    promptTokens, completionTokens, totalTokens);
        }
    }

    // ========== 核心方法 ==========

    /**
     * 发起流式请求
     *
     * @param model        模型 ID
     * @param systemPrompt System 消息内容
     * @param userMessage  用户消息内容
     * @param callback     事件回调
     */
    public static void streamChat(String model, String systemPrompt,
                                  String userMessage, StreamCallback callback) {
        // 1. 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", model);
        requestBody.addProperty("temperature", 0.7);
        requestBody.addProperty("max_tokens", 2048);
        requestBody.addProperty("stream", true);

        JsonArray messages = new JsonArray();
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            JsonObject sysMsg = new JsonObject();
            sysMsg.addProperty("role", "system");
            sysMsg.addProperty("content", systemPrompt);
            messages.add(sysMsg);
        }
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", userMessage);
        messages.add(userMsg);
        requestBody.add("messages", messages);

        // 2. 创建 HTTP 客户端
        // 关键：readTimeout 是"两个数据块之间的最大等待时间"，不是整个响应的超时
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(15, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)   // 流式场景需要更长
                .writeTimeout(15, TimeUnit.SECONDS)
                .build();

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .addHeader("Accept", "text/event-stream")  // 明确告诉服务端我要 SSE
                .post(RequestBody.create(requestBody.toString(),
                        MediaType.parse("application/json")))
                .build();

        // 3. 发起请求并解析 SSE 流
        StringBuilder fullContent = new StringBuilder();
        Usage usage = null;

        try (Response response = client.newCall(request).execute()) {
            // 检查 HTTP 状态码
            if (!response.isSuccessful()) {
                String errorBody = response.body() != null ? response.body().string() : "无响应体";
                callback.onError(
                        new RuntimeException("HTTP " + response.code() + ": " + errorBody),
                        fullContent.toString()
                );
                return;
            }

            // 逐行读取 SSE 流（显式指定 UTF-8，SSE 规范要求 UTF-8 编码）
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(response.body().byteStream(), StandardCharsets.UTF_8));
            String line;
            boolean streamDone = false;  // 是否收到了 [DONE] 标记

            while ((line = reader.readLine()) != null) {
                // 跳过空行（SSE 事件分隔符）
                if (line.isEmpty()) {
                    continue;
                }

                // 跳过注释行（心跳保活）
                if (line.startsWith(":")) {
                    continue;
                }

                // 只处理 data: 开头的行（兼容 "data: xxx" 和 "data:xxx" 两种格式）
                if (!line.startsWith("data:")) {
                    continue;
                }

                // 去掉 "data:" 前缀，SSE 标准规定冒号后最多去掉一个可选空格
                String data = line.substring(5);
                if (data.startsWith(" ")) {
                    data = data.substring(1);
                }

                // 检查流结束标记
                if ("[DONE]".equals(data)) {
                    streamDone = true;
                    break;
                }

                // 解析 JSON（加容错）
                JsonObject chunk;
                try {
                    chunk = JsonParser.parseString(data).getAsJsonObject();
                } catch (Exception e) {
                    // JSON 解析失败，跳过这个 chunk，不要中断整个流
                    System.err.println("JSON 解析失败，跳过: " + data);
                    continue;
                }

                // 提取 choices 数组
                JsonArray choices = chunk.getAsJsonArray("choices");
                if (choices == null || choices.isEmpty()) {
                    // 有些平台在最后一个 chunk（stream_options 模式）choices 为空数组
                    // 但可能有 usage 字段
                    usage = extractUsage(chunk, usage);
                    continue;
                }

                JsonObject choice = choices.get(0).getAsJsonObject();

                // 提取 delta 中的 content
                JsonObject delta = choice.getAsJsonObject("delta");
                if (delta != null && delta.has("content")) {
                    JsonElement contentElement = delta.get("content");
                    if (!contentElement.isJsonNull()) {
                        String token = contentElement.getAsString();
                        if (!token.isEmpty()) {
                            fullContent.append(token);
                            callback.onToken(token);
                        }
                    }
                }

                // 提取 finish_reason
                JsonElement finishElement = choice.get("finish_reason");
                if (finishElement != null && !finishElement.isJsonNull()) {
                    String finishReason = finishElement.getAsString();
                    // finish_reason 不只是 "stop"，还可能是：
                    // - "length"：达到 max_tokens 上限，内容被截断
                    // - "content_filter"：被安全过滤截断
                    // - "tool_calls"：模型转入工具调用流程
                    // 这里统一标记为流结束，调用方可根据 finishReason 做更细的处理
                    usage = extractUsage(chunk, usage);
                }
            }

            // 判断流是否正常结束
            if (streamDone) {
                callback.onComplete(fullContent.toString(), usage);
            } else {
                // readLine() 返回 null 但没收到 [DONE]——连接异常关闭
                callback.onError(
                        new RuntimeException("SSE 流异常结束：未收到 [DONE] 标记"),
                        fullContent.toString()
                );
            }

        } catch (Exception e) {
            // 连接异常（超时、网络中断等），把已接收到的内容传给调用方
            callback.onError(e, fullContent.toString());
        }
    }

    /**
     * 从 chunk 中提取 usage 信息
     */
    private static Usage extractUsage(JsonObject chunk, Usage existing) {
        if (!chunk.has("usage") || chunk.get("usage").isJsonNull()) {
            return existing;
        }
        JsonObject usageJson = chunk.getAsJsonObject("usage");
        Usage usage = new Usage();
        usage.promptTokens = usageJson.has("prompt_tokens")
                ? usageJson.get("prompt_tokens").getAsInt() : 0;
        usage.completionTokens = usageJson.has("completion_tokens")
                ? usageJson.get("completion_tokens").getAsInt() : 0;
        usage.totalTokens = usageJson.has("total_tokens")
                ? usageJson.get("total_tokens").getAsInt() : 0;
        return usage;
    }

    // ========== 运行示例 ==========

    public static void main(String[] args) {
        System.out.println("=== SSE 流式调用演示 ===\n");

        streamChat(
                "Qwen/Qwen3-32B",
                "你是一个技术专家，回答简洁清晰。",
                "用两三句话解释一下什么是 SSE 协议？",
                new StreamCallback() {
                    @Override
                    public void onToken(String token) {
                        // 每收到一个 token 就实时输出（不换行）
                        System.out.print(token);
                    }

                    @Override
                    public void onComplete(String fullContent, Usage usage) {
                        System.out.println("\n");
                        System.out.println("--- 流式输出完毕 ---");
                        System.out.println("完整内容长度：" + fullContent.length() + " 字符");
                        if (usage != null) {
                            System.out.println("Token 统计：" + usage);
                        } else {
                            System.out.println("Token 统计：未返回");
                        }
                    }

                    @Override
                    public void onError(Exception e, String partialContent) {
                        System.err.println("\n\n--- 发生错误 ---");
                        System.err.println("错误信息：" + e.getMessage());
                        if (!partialContent.isEmpty()) {
                            System.err.println("已接收到的内容：" + partialContent);
                        }
                    }
                }
        );
    }
}
