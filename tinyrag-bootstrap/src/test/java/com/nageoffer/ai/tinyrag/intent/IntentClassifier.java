package com.nageoffer.ai.tinyrag.intent;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.List;
import java.util.Set;

public class IntentClassifier {

    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "your-api-key";
    // 意图分类用小模型就够了
    private static final String MODEL = "deepseek-ai/DeepSeek-V3.2";
    private static final OkHttpClient client = new OkHttpClient();
    private static final Gson gson = new Gson();

    /**
     * 意图分类 Prompt 模板
     */
    private static final String CLASSIFY_PROMPT = """
            你是一个意图分类助手。根据对话历史和用户的最新消息，判断用户的意图类别。
            
            意图类别定义：
            1. knowledge - 知识检索：用户在询问产品信息、政策规定、操作指南等通用知识。
               示例："iPhone 16 Pro 的退货政策是什么""保修期多久""配送范围覆盖哪些城市"
            2. tool - 工具调用：用户想查询个人数据、实时信息，或执行某个操作。
               示例："查一下我的订单状态""帮我申请退货""我还剩几天年假"
            3. chitchat - 闲聊对话：用户在打招呼、感谢、闲聊，不涉及具体业务问题。
               示例："你好""谢谢""你是AI吗""今天心情不好"
            4. clarification - 引导澄清：用户的问题太模糊，缺少关键信息，无法确定意图。
               示例："有什么推荐的""怎么办""帮我看看"
            
            判断规则：
            - 结合对话历史判断，相同的话在不同上下文中意图可能不同
            - 如果用户的问题涉及"我的""查一下"等个人化表述，通常是工具调用
            - 如果问题在问通用的规则、政策、产品信息，通常是知识检索
            - 只有在真的无法判断意图时才分类为 clarification
            - 以 JSON 格式输出：{"intent": "分类结果", "confidence": 置信度}
            - 不要输出 JSON 以外的任何内容
            
            对话历史：
            %s
            
            用户最新消息：%s""";

    /**
     * 闲聊关键词（规则层快速过滤）
     */
    private static final Set<String> CHITCHAT_KEYWORDS = Set.of(
            "你好", "您好", "谢谢", "感谢", "再见", "拜拜",
            "哈哈", "嗯嗯", "好的", "收到", "明白了", "ok"
    );

    /**
     * 混合方案：规则优先 + 大模型兜底
     */
    public static IntentResult classify(List<Message> history,
                                        String query) throws IOException {
        // 第一层：规则快速过滤
        if (query.length() <= 6 && CHITCHAT_KEYWORDS.contains(query)) {
            return new IntentResult("chitchat", 0.99);
        }

        // 第二层：大模型分类
        return classifyByLLM(history, query);
    }

    /**
     * 调用大模型做意图分类
     */
    private static IntentResult classifyByLLM(List<Message> history,
                                              String query) throws IOException {
        // 构建对话历史文本
        StringBuilder historyText = new StringBuilder();
        if (history.isEmpty()) {
            historyText.append("（无历史对话）");
        } else {
            for (Message msg : history) {
                String roleName = "user".equals(msg.role) ? "用户" : "助手";
                historyText.append(roleName).append("：")
                        .append(msg.content).append("\n");
            }
        }

        // 构建分类请求
        String prompt = String.format(CLASSIFY_PROMPT,
                historyText, query);

        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("temperature", 0.1);
        body.addProperty("max_tokens", 100);
        JsonArray messages = new JsonArray();
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", prompt);
        messages.add(userMsg);
        body.add("messages", messages);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(body.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            assert response.body() != null;
            String responseBody = response.body().string();
            JsonObject json = gson.fromJson(responseBody, JsonObject.class);
            String content = json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString().trim();

            // 解析 JSON 结果
            return parseIntentResult(content);
        }
    }

    /**
     * 解析模型返回的意图分类 JSON
     */
    private static IntentResult parseIntentResult(String content) {
        try {
            JsonObject result = gson.fromJson(content, JsonObject.class);
            String intent = result.get("intent").getAsString();
            double confidence = result.has("confidence")
                    ? result.get("confidence").getAsDouble() : 0.8;

            // 校验意图是否合法
            Set<String> validIntents = Set.of(
                    "knowledge", "tool", "chitchat", "clarification");
            if (!validIntents.contains(intent)) {
                return new IntentResult("knowledge", 0.5);
            }

            return new IntentResult(intent, confidence);
        } catch (Exception e) {
            // JSON 解析失败，兜底走知识检索
            return new IntentResult("knowledge", 0.5);
        }
    }

    /**
     * 意图分类结果
     */
    public static class IntentResult {
        public String intent;
        public double confidence;

        public IntentResult(String intent, double confidence) {
            this.intent = intent;
            this.confidence = confidence;
        }

        @Override
        public String toString() {
            return String.format("{intent: \"%s\", confidence: %.2f}",
                    intent, confidence);
        }
    }

    /**
     * 消息数据结构
     */
    public static class Message {
        public String role;
        public String content;

        public Message(String role, String content) {
            this.role = role;
            this.content = content;
        }
    }

    public static void main(String[] args) throws IOException {
        // ===== 场景 1：知识检索 =====
        System.out.println("===== 场景 1：知识检索 =====");
        List<Message> history1 = List.of();
        String query1 = "iPhone 16 Pro 的退货政策是什么？";
        IntentResult result1 = IntentClassifier.classify(history1, query1);
        System.out.println("用户消息：" + query1);
        System.out.println("意图分类：" + result1);
        System.out.println("路由结果：" + IntentRouter.route(result1, history1, query1));

        // ===== 场景 2：工具调用 =====
        System.out.println("\n===== 场景 2：工具调用 =====");
        List<Message> history2 = List.of();
        String query2 = "帮我查一下订单 #12345 的物流状态";
        IntentResult result2 = IntentClassifier.classify(history2, query2);
        System.out.println("用户消息：" + query2);
        System.out.println("意图分类：" + result2);
        System.out.println("路由结果：" + IntentRouter.route(result2, history2, query2));

        // ===== 场景 3：闲聊（规则层命中） =====
        System.out.println("\n===== 场景 3：闲聊（规则命中） =====");
        List<Message> history3 = List.of();
        String query3 = "你好";
        IntentResult result3 = IntentClassifier.classify(history3, query3);
        System.out.println("用户消息：" + query3);
        System.out.println("意图分类：" + result3);
        System.out.println("路由结果：" + IntentRouter.route(result3, history3, query3));

        // ===== 场景 4：引导澄清 =====
        System.out.println("\n===== 场景 4：引导澄清 =====");
        List<Message> history4 = List.of();
        String query4 = "有什么推荐的吗？";
        IntentResult result4 = IntentClassifier.classify(history4, query4);
        System.out.println("用户消息：" + query4);
        System.out.println("意图分类：" + result4);
        System.out.println("路由结果：" + IntentRouter.route(result4, history4, query4));

        // ===== 场景 5：上下文相关的意图判断 =====
        System.out.println("\n===== 场景 5：上下文相关 =====");
        List<Message> history5 = List.of(
                new Message("user", "iPhone 16 Pro 的退货政策是什么？"),
                new Message("assistant",
                        "iPhone 16 Pro 支持七天无理由退货，拆封后如有质量问题可联系售后。"),
                new Message("user", "我买的那台屏幕有亮点，想退货"),
                new Message("assistant",
                        "屏幕亮点属于质量问题，可以申请退货。请问您需要我帮您发起退货申请吗？")
        );
        String query5 = "好的，帮我退了吧";
        IntentResult result5 = IntentClassifier.classify(history5, query5);
        System.out.println("对话历史：（用户在聊 iPhone 16 Pro 退货，助手问是否发起退货申请）");
        System.out.println("用户消息：" + query5);
        System.out.println("意图分类：" + result5);
        System.out.println("路由结果：" + IntentRouter.route(result5, history5, query5));
    }
}