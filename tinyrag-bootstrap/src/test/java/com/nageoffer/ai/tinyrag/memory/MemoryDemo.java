package com.nageoffer.ai.tinyrag.memory;

import com.google.gson.*;
import okhttp3.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MemoryDemo {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "your-api-key";
    private static final String MODEL = "Qwen/Qwen3-8B";
    private static final OkHttpClient client = new OkHttpClient();
    private static final Gson gson = new Gson();

    public static void main(String[] args) throws IOException {
        System.out.println("===== 无记忆模式 =====");
        noMemoryDemo();

        System.out.println("\n===== 有记忆模式 =====");
        withMemoryDemo();
    }

    /**
     * 无记忆模式：每次请求只带当前问题，不带历史消息
     */
    static void noMemoryDemo() throws IOException {
        // 第 1 轮
        String answer1 = chat(List.of(
                message("system", "你是一个电商客服助手，简洁回答用户问题。"),
                message("user", "iPhone 16 Pro 的退货政策是什么？")
        ));
        System.out.println("用户：iPhone 16 Pro 的退货政策是什么？");
        System.out.println("助手：" + answer1);

        // 第 2 轮：不带历史消息，模型不知道"它"是什么
        String answer2 = chat(List.of(
                message("system", "你是一个电商客服助手，简洁回答用户问题。"),
                message("user", "那它的保修期呢？")
        ));
        System.out.println("\n用户：那它的保修期呢？");
        System.out.println("助手：" + answer2);
    }

    /**
     * 有记忆模式：每次请求带上完整的历史消息
     */
    static void withMemoryDemo() throws IOException {
        List<JsonObject> history = new ArrayList<>();
        history.add(message("system", "你是一个电商客服助手，简洁回答用户问题。"));

        // 第 1 轮
        history.add(message("user", "iPhone 16 Pro 的退货政策是什么？"));
        String answer1 = chat(history);
        history.add(message("assistant", answer1));
        System.out.println("用户：iPhone 16 Pro 的退货政策是什么？");
        System.out.println("助手：" + answer1);

        // 第 2 轮：带上第 1 轮的历史，模型知道"它"指 iPhone 16 Pro
        history.add(message("user", "那它的保修期呢？"));
        String answer2 = chat(history);
        history.add(message("assistant", answer2));
        System.out.println("\n用户：那它的保修期呢？");
        System.out.println("助手：" + answer2);

        // 第 3 轮：继续追问
        history.add(message("user", "过了保修期维修大概多少钱？"));
        String answer3 = chat(history);
        System.out.println("\n用户：过了保修期维修大概多少钱？");
        System.out.println("助手：" + answer3);
    }

    /**
     * 调用 SiliconFlow Chat API
     */
    static String chat(List<JsonObject> messages) throws IOException {
        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("temperature", 0.1);
        body.addProperty("max_tokens", 512);
        JsonArray messagesArray = new JsonArray();
        for (JsonObject msg : messages) {
            messagesArray.add(msg);
        }
        body.add("messages", messagesArray);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(body.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            String responseBody = response.body().string();
            JsonObject json = gson.fromJson(responseBody, JsonObject.class);
            return json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        }
    }

    static JsonObject message(String role, String content) {
        JsonObject msg = new JsonObject();
        msg.addProperty("role", role);
        msg.addProperty("content", content);
        return msg;
    }
}
