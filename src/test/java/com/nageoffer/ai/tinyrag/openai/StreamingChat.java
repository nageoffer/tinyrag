package com.nageoffer.ai.tinyrag.openai;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import okhttp3.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;

public class StreamingChat {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "YOUR_API_KEY";

    public static void main(String[] args) throws IOException {
        // 1. 构建请求体（注意 stream 设为 true）
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", "Qwen/Qwen3-32B");
        requestBody.addProperty("temperature", 0.1);
        requestBody.addProperty("max_tokens", 1024);
        requestBody.addProperty("stream", true);  // 开启流式

        JsonArray messages = new JsonArray();

        JsonObject systemMsg = new JsonObject();
        systemMsg.addProperty("role", "system");
        systemMsg.addProperty("content", "你是一个专业的电商客服助手，回答要简洁明了。");
        messages.add(systemMsg);

        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", "买了一周的东西还能退吗？");
        messages.add(userMsg);

        requestBody.add("messages", messages);

        // 2. 创建 OkHttp 客户端
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(120, TimeUnit.SECONDS)  // 流式调用需要更长的读取超时
                .build();

        // 3. 构建请求
        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        requestBody.toString(),
                        MediaType.parse("application/json")
                ))
                .build();

        // 4. 发送请求并逐行读取 SSE 响应
        Gson gson = new Gson();
        StringBuilder fullContent = new StringBuilder();

        System.out.println("=== 模型回答（流式输出）===");

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                System.out.println("请求失败，状态码：" + response.code());
                System.out.println("错误信息：" + response.body().string());
                return;
            }

            // 逐行读取响应体
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(response.body().byteStream())
            );

            String line;
            while ((line = reader.readLine()) != null) {
                // 跳过空行
                if (line.isEmpty()) {
                    continue;
                }

                // 每行以 "data: " 开头，去掉前缀
                if (!line.startsWith("data: ")) {
                    continue;
                }
                String data = line.substring(6);  // 去掉 "data: " 前缀（6 个字符）

                // 检查是否是结束标记
                if ("[DONE]".equals(data)) {
                    break;
                }

                // 解析 JSON，提取增量内容
                JsonObject chunk = gson.fromJson(data, JsonObject.class);
                JsonArray choices = chunk.getAsJsonArray("choices");
                if (choices != null && choices.size() > 0) {
                    JsonObject delta = choices.get(0).getAsJsonObject()
                            .getAsJsonObject("delta");
                    if (delta != null && delta.has("content")) {
                        JsonElement contentElement = delta.get("content");
                        if (!contentElement.isJsonNull()) {
                            String content = contentElement.getAsString();
                            // 实时打印增量内容（不换行，模拟打字效果）
                            System.out.print(content);
                            fullContent.append(content);
                        }
                    }
                }
            }
        }

        // 输出完毕，换行
        System.out.println();
        System.out.println();
        System.out.println("=== 完整回答 ===");
        System.out.println(fullContent);
    }
}