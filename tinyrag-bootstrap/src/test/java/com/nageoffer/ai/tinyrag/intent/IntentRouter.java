package com.nageoffer.ai.tinyrag.intent;

import java.io.IOException;
import java.util.List;

public class IntentRouter {

    /**
     * 路由到对应的处理流程
     */
    public static String route(IntentClassifier.IntentResult intentResult,
                               List<IntentClassifier.Message> history,
                               String query) throws IOException {
        return switch (intentResult.intent) {
            case "tool" -> handleTool(history, query);
            case "chitchat" -> handleChitchat(query);
            case "clarification" -> handleClarification(query);
            default -> handleKnowledge(history, query);
        };
    }

    /**
     * 知识检索路径
     * 实际项目中：Query 改写 → 向量检索 → 重排序 → 生成答案
     */
    private static String handleKnowledge(List<IntentClassifier.Message> history,
                                          String query) {
        // 这里简化处理，实际项目中要走完整的 RAG 流程
        return "[知识检索] 正在从知识库检索「" + query + "」相关内容并生成答案...";
    }

    /**
     * 工具调用路径
     * 实际项目中：识别工具 → 传入参数 → 执行 → 返回结果
     */
    private static String handleTool(List<IntentClassifier.Message> history,
                                     String query) {
        // 这里简化处理，实际项目中要走 Function Call / MCP 流程
        return "[工具调用] 正在调用相关工具处理「" + query + "」...";
    }

    /**
     * 闲聊路径
     * 直接调模型生成回复，不带检索上下文
     */
    private static String handleChitchat(String query) {
        // 简单的闲聊回复映射，实际项目中可以调模型生成
        if (query.contains("你好") || query.contains("您好")) {
            return "[闲聊] 您好！请问有什么可以帮您的？";
        }
        if (query.contains("谢谢") || query.contains("感谢")) {
            return "[闲聊] 不客气，还有其他问题随时问我。";
        }
        return "[闲聊] 好的，如果您有任何产品或服务相关的问题，随时告诉我。";
    }

    /**
     * 引导澄清路径
     * 返回引导性问题，让用户补充信息
     */
    private static String handleClarification(String query) {
        return "[引导澄清] 您的问题我还不太明确，"
                + "能否告诉我您想了解哪方面的信息？"
                + "比如：产品信息、订单查询、退换货政策等。";
    }
}
