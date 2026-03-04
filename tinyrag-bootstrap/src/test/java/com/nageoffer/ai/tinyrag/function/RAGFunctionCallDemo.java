package com.nageoffer.ai.tinyrag.function;

import cn.hutool.core.collection.CollUtil;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;

public class RAGFunctionCallDemo {

    private static final String API_KEY = "YOUR_API_KEY";
    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String MODEL = "deepseek-ai/DeepSeek-V3";

    private static final OkHttpClient client = new OkHttpClient();
    private static final Gson gson = new GsonBuilder().setPrettyPrinting().create();

    public static void main(String[] args) throws IOException {
        // 用户问题
        String userQuestion = "我还剩几天年假";
        System.out.println("用户问题：" + userQuestion);
        System.out.println("\n" + "=".repeat(60) + "\n");

        // 第一轮：发送工具列表和用户问题
        JsonObject firstResponse = callModelWithTools(userQuestion);
        System.out.println("第一轮响应：");
        System.out.println(gson.toJson(firstResponse));
        System.out.println("\n" + "=".repeat(60) + "\n");

        // 解析 tool_calls
        JsonArray toolCalls = firstResponse.getAsJsonArray("choices")
                .get(0).getAsJsonObject()
                .getAsJsonObject("message")
                .getAsJsonArray("tool_calls");

        if (CollUtil.isEmpty(toolCalls)) {
            System.out.println("模型没有调用工具，直接返回答案");
            return;
        }

        // 执行函数
        JsonObject toolCall = toolCalls.get(0).getAsJsonObject();
        String functionName = toolCall.getAsJsonObject("function").get("name").getAsString();
        String arguments = toolCall.getAsJsonObject("function").get("arguments").getAsString();
        String toolCallId = toolCall.get("id").getAsString();

        System.out.println("模型要调用的函数：" + functionName);
        System.out.println("函数参数：" + arguments);
        System.out.println("\n" + "=".repeat(60) + "\n");

        // 执行函数（这里用 mock 数据）
        String functionResult = executeFunction(functionName, arguments);
        System.out.println("函数执行结果：" + functionResult);
        System.out.println("\n" + "=".repeat(60) + "\n");

        // 第二轮：把结果返回给模型
        JsonObject secondResponse = callModelWithFunctionResult(
                userQuestion, toolCall, toolCallId, functionResult);

        System.out.println("第二轮响应：");
        System.out.println(gson.toJson(secondResponse));
        System.out.println("\n" + "=".repeat(60) + "\n");

        // 提取最终答案
        String finalAnswer = secondResponse.getAsJsonArray("choices")
                .get(0).getAsJsonObject()
                .getAsJsonObject("message")
                .get("content").getAsString();

        System.out.println("最终答案：" + finalAnswer);
    }

    /**
     * 第一轮调用：发送工具列表和用户问题
     */
    private static JsonObject callModelWithTools(String userQuestion) throws IOException {
        // 定义工具列表
        JsonArray tools = new JsonArray();

        // 工具 1：查询年假余额
        JsonObject tool1 = new JsonObject();
        tool1.addProperty("type", "function");
        JsonObject function1 = new JsonObject();
        function1.addProperty("name", "getUserAnnualLeave");
        function1.addProperty("description", "查询用户的年假余额，包括总天数、已使用天数、剩余天数");
        JsonObject parameters1 = new JsonObject();
        parameters1.addProperty("type", "object");
        JsonObject properties1 = new JsonObject();
        JsonObject userId1 = new JsonObject();
        userId1.addProperty("type", "string");
        userId1.addProperty("description", "用户 ID");
        properties1.add("userId", userId1);
        parameters1.add("properties", properties1);
        JsonArray required1 = new JsonArray();
        required1.add("userId");
        parameters1.add("required", required1);
        function1.add("parameters", parameters1);
        tool1.add("function", function1);
        tools.add(tool1);

        // 工具 2：查询订单状态
        JsonObject tool2 = new JsonObject();
        tool2.addProperty("type", "function");
        JsonObject function2 = new JsonObject();
        function2.addProperty("name", "getOrderStatus");
        function2.addProperty("description", "查询订单的物流状态和详细信息");
        JsonObject parameters2 = new JsonObject();
        parameters2.addProperty("type", "object");
        JsonObject properties2 = new JsonObject();
        JsonObject orderId = new JsonObject();
        orderId.addProperty("type", "string");
        orderId.addProperty("description", "订单号");
        properties2.add("orderId", orderId);
        parameters2.add("properties", properties2);
        JsonArray required2 = new JsonArray();
        required2.add("orderId");
        parameters2.add("required", required2);
        function2.add("parameters", parameters2);
        tool2.add("function", function2);
        tools.add(tool2);

        // 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", MODEL);

        JsonArray messages = new JsonArray();

        // ★★★ 新增：添加系统消息，带上用户ID ★★★
        JsonObject systemMessage = new JsonObject();
        systemMessage.addProperty("role", "system");
        systemMessage.addProperty("content", "当前登录用户的ID是: user_12345");
        messages.add(systemMessage);

        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", userQuestion);
        messages.add(userMessage);

        requestBody.add("messages", messages);
        requestBody.add("tools", tools);
        requestBody.addProperty("tool_choice", "auto");

        // 发送请求
        return sendRequest(requestBody);
    }

    /**
     * 第二轮调用：把函数执行结果返回给模型
     */
    private static JsonObject callModelWithFunctionResult(
            String userQuestion, JsonObject toolCall, String toolCallId, String functionResult) throws IOException {

        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", MODEL);

        JsonArray messages = new JsonArray();

        // 第一条消息：用户问题
        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", userQuestion);
        messages.add(userMessage);

        // 第二条消息：第一轮的模型响应（带 tool_calls）
        JsonObject assistantMessage = new JsonObject();
        assistantMessage.addProperty("role", "assistant");
        assistantMessage.add("content", JsonNull.INSTANCE);
        JsonArray toolCalls = new JsonArray();
        toolCalls.add(toolCall);
        assistantMessage.add("tool_calls", toolCalls);
        messages.add(assistantMessage);

        // 第三条消息：函数执行结果
        JsonObject toolMessage = new JsonObject();
        toolMessage.addProperty("role", "tool");
        toolMessage.addProperty("tool_call_id", toolCallId);
        toolMessage.addProperty("content", functionResult);
        messages.add(toolMessage);

        requestBody.add("messages", messages);

        // 发送请求
        return sendRequest(requestBody);
    }

    /**
     * 执行函数（这里用 mock 数据模拟）
     */
    private static String executeFunction(String functionName, String arguments) {
        JsonObject args = gson.fromJson(arguments, JsonObject.class);

        if ("getUserAnnualLeave".equals(functionName)) {
            // 模拟查询 HR 系统
            String userId = args.get("userId").getAsString();
            JsonObject result = new JsonObject();
            result.addProperty("userId", userId);
            result.addProperty("remainingDays", 5);
            result.addProperty("totalDays", 10);
            result.addProperty("usedDays", 5);
            return gson.toJson(result);
        } else if ("getOrderStatus".equals(functionName)) {
            // 模拟查询订单系统
            String orderId = args.get("orderId").getAsString();
            JsonObject result = new JsonObject();
            result.addProperty("orderId", orderId);
            result.addProperty("status", "运输中");
            result.addProperty("location", "北京市朝阳区分拨中心");
            result.addProperty("estimatedDelivery", "2026-02-28");
            return gson.toJson(result);
        }

        return "{\"error\": \"未知的函数\"}";
    }

    /**
     * 发送 HTTP 请求
     */
    private static JsonObject sendRequest(JsonObject requestBody) throws IOException {
        RequestBody body = RequestBody.create(
                gson.toJson(requestBody),
                MediaType.parse("application/json"));

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("请求失败：" + response);
            }
            String responseBody = response.body().string();
            return gson.fromJson(responseBody, JsonObject.class);
        }
    }
}
