package com.nageoffer.ai.tinyrag.memory;

import com.google.gson.*;
import okhttp3.*;
import java.io.IOException;
import java.util.*;

/**
 * 支持摘要压缩的会话记忆管理器
 */
public class SummaryMemory {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "sk-jppuymtrzvsqylydmlvolexkxiibehqumnzvvpkchagwllip";
    private static final String MODEL = "Qwen/Qwen3-8B";
    private static final OkHttpClient client = new OkHttpClient();
    private static final Gson gson = new Gson();

    /** 触发摘要的 Token 阈值 */
    private final int tokenThreshold;
    /** 保留最近的完整对话轮数 */
    private final int keepRecentRounds;

    /** 会话存储 */
    private final Map<String, List<JsonObject>> store = new HashMap<>();
    /** 摘要存储 */
    private final Map<String, String> summaryStore = new HashMap<>();

    public SummaryMemory(int tokenThreshold, int keepRecentRounds) {
        this.tokenThreshold = tokenThreshold;
        this.keepRecentRounds = keepRecentRounds;
    }

    public void addMessage(String sessionId, String role, String content) {
        store.computeIfAbsent(sessionId, k -> new ArrayList<>())
                .add(message(role, content));

        // 检查是否需要触发摘要压缩
        int totalTokens = estimateTotalTokens(sessionId);
        if (totalTokens > tokenThreshold) {
            try {
                compress(sessionId);
            } catch (IOException e) {
                System.err.println("摘要压缩失败：" + e.getMessage());
            }
        }
    }

    /**
     * 压缩早期对话为摘要
     */
    private void compress(String sessionId) throws IOException {
        List<JsonObject> allMessages = store.get(sessionId);
        if (allMessages == null || allMessages.size() <= keepRecentRounds * 2) {
            return;
        }

        // 分离：早期消息（要压缩的）+ 最近消息（要保留的）
        int keepCount = keepRecentRounds * 2;
        List<JsonObject> earlyMessages = allMessages.subList(
                0, allMessages.size() - keepCount);
        List<JsonObject> recentMessages = new ArrayList<>(
                allMessages.subList(allMessages.size() - keepCount, allMessages.size()));

        // 构建要压缩的对话文本
        StringBuilder conversationText = new StringBuilder();
        for (JsonObject msg : earlyMessages) {
            String role = msg.get("role").getAsString();
            String content = msg.get("content").getAsString();
            conversationText.append(role).append("：").append(content).append("\n");
        }

        // 获取已有的摘要
        String existingSummary = summaryStore.getOrDefault(sessionId, "");

        // 调用大模型生成摘要
        String summaryPrompt = "请将以下对话历史压缩为一段简洁的摘要，要求：\n" +
                "1. 保留用户的核心意图和关注点\n" +
                "2. 保留所有关键实体（产品名、订单号、日期、金额等）\n" +
                "3. 保留已经确认的结论和决定\n" +
                "4. 保留尚未解决的问题\n" +
                "5. 省略寒暄、重复确认、无关细节\n" +
                "6. 摘要以第三人称描述，控制在 200 字以内\n";

        if (!existingSummary.isEmpty()) {
            summaryPrompt += "\n已有的历史摘要：\n" + existingSummary + "\n";
        }
        summaryPrompt += "\n需要压缩的新对话：\n" + conversationText;

        String summary = chat(List.of(
                message("system", "你是一个对话摘要助手，负责将对话历史压缩为简洁的摘要。"),
                message("user", summaryPrompt)
        ));

        // 更新摘要和消息列表
        summaryStore.put(sessionId, summary);
        store.put(sessionId, recentMessages);

        System.out.println("[摘要压缩] 将 " + earlyMessages.size() +
                " 条早期消息压缩为摘要");
        System.out.println("[摘要内容] " + summary);
    }

    /**
     * 构建发送给 API 的完整 messages 数组
     */
    public List<JsonObject> buildMessages(String sessionId,
                                          String systemPrompt,
                                          String currentQuestion) {
        List<JsonObject> messages = new ArrayList<>();
        messages.add(message("system", systemPrompt));

        // 添加摘要（如果有）
        String summary = summaryStore.get(sessionId);
        if (summary != null && !summary.isEmpty()) {
            messages.add(message("system",
                    "【对话背景摘要】" + summary));
        }

        // 添加最近的完整对话
        List<JsonObject> recentMessages = store.getOrDefault(
                sessionId, List.of());
        messages.addAll(recentMessages);

        // 添加当前问题
        messages.add(message("user", currentQuestion));
        return messages;
    }

    private int estimateTotalTokens(String sessionId) {
        List<JsonObject> messages = store.getOrDefault(sessionId, List.of());
        int total = 0;
        for (JsonObject msg : messages) {
            total += estimateTokens(msg.get("content").getAsString());
        }
        return total;
    }

    /** 简单的 Token 估算 */
    static int estimateTokens(String text) {
        if (text == null || text.isEmpty()) return 0;
        int chineseChars = 0, otherChars = 0;
        for (char c : text.toCharArray()) {
            if (Character.UnicodeScript.of(c) == Character.UnicodeScript.HAN) {
                chineseChars++;
            } else if (!Character.isWhitespace(c)) {
                otherChars++;
            }
        }
        return (int) (chineseChars * 1.5 + otherChars / 4.0);
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
