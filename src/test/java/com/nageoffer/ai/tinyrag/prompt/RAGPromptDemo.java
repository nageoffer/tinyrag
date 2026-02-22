package com.nageoffer.ai.tinyrag.prompt;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class RAGPromptDemo {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "YOUR_API_KEY";

    /**
     * 调用大模型 API
     *
     * @param systemPrompt 系统提示词（可选，这里我们把所有规则都放在 user message 里了）
     * @param userMessage  用户消息（包含参考资料和用户问题）
     * @return 模型回答
     */
    public static String callLLM(String systemPrompt, String userMessage) throws IOException {
        // 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", "Qwen/Qwen3-32B");
        requestBody.addProperty("temperature", 0.1);  // RAG 场景推荐低温度
        requestBody.addProperty("max_tokens", 1024);
        requestBody.addProperty("stream", false);

        JsonArray messages = new JsonArray();

        // 如果有 system prompt，加上
        if (systemPrompt != null && !systemPrompt.isEmpty()) {
            JsonObject systemMsg = new JsonObject();
            systemMsg.addProperty("role", "system");
            systemMsg.addProperty("content", systemPrompt);
            messages.add(systemMsg);
        }

        // user message
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", userMessage);
        messages.add(userMsg);

        requestBody.add("messages", messages);

        // 创建 HTTP 客户端
        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();

        // 构建请求
        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        requestBody.toString(),
                        MediaType.parse("application/json")
                ))
                .build();

        // 发送请求
        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("请求失败，状态码：" + response.code());
            }

            String responseBody = response.body().string();
            Gson gson = new Gson();
            JsonObject jsonResponse = gson.fromJson(responseBody, JsonObject.class);

            // 提取模型回答
            return jsonResponse
                    .getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        }
    }

    public static void main(String[] args) throws IOException {
        // 模拟检索到的 chunk
        List<Chunk> chunks = new ArrayList<>();
        chunks.add(new Chunk(
                "1",
                "《退货政策》",
                "2025-01-15",
                "自签收之日起 7 天内，商品未使用且不影响二次销售的，可以申请七天无理由退货。"
        ));
        chunks.add(new Chunk(
                "2",
                "《运费说明》",
                "2025-01-10",
                "七天无理由退货的运费由买家承担。"
        ));

        // 用户问题
        String question = "买了一周的东西还能退吗？";

        // 组装 Prompt
        String userMessage = RAGPromptTemplate.buildPrompt(chunks, question);

        // 调用 API
        String answer = callLLM(null, userMessage);

        // 输出结果
        System.out.println("=== 用户问题 ===");
        System.out.println(question);
        System.out.println();
        System.out.println("=== 模型回答 ===");
        System.out.println(answer);
    }
}