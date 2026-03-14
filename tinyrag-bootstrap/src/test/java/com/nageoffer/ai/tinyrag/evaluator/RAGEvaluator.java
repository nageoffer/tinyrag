package com.nageoffer.ai.tinyrag.evaluator;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import okhttp3.*;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class RAGEvaluator {

    private static final String API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions";
    private static final String API_KEY = "your_apikey"; // 替换为你的 API Key 这里用的阿里云百炼平台
    private static final String JUDGE_MODEL = "qwen3.5-flash";
    private static final Gson gson = new Gson();
    private static final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(120, TimeUnit.SECONDS)
            .build();

    // 评测数据结构
    static class EvalCase {
        String query;            // 用户问题
        String expectedAnswer;   // 标准答案
        List<String> relevantChunkIds;  // 正确答案对应的 chunk ID
        String intent;           // 意图类别

        EvalCase(String query, String expectedAnswer, List<String> relevantChunkIds, String intent) {
            this.query = query;
            this.expectedAnswer = expectedAnswer;
            this.relevantChunkIds = relevantChunkIds;
            this.intent = intent;
        }
    }

    // 评分结果
    static class ScoreResult {
        int score;       // 1-5 分
        String label;    // faithful / relevant / correct 等
        String reason;   // 评分理由
    }

    // 单条评测结果
    static class EvalResult {
        EvalCase evalCase;
        List<String> retrievedChunkIds;  // 实际检索到的 chunk ID 列表
        String actualAnswer;             // 模型实际生成的答案
        boolean hit;                     // 检索是否命中
        double reciprocalRank;           // 倒数排名
        ScoreResult faithfulness;        // 忠实度评分
        ScoreResult relevancy;           // 相关性评分
        ScoreResult correctness;         // 正确率评分
    }

    // 构建评测数据集
    static List<EvalCase> buildEvalDataset() {
        return List.of(
                new EvalCase(
                        "iPhone 16 Pro 的退货政策是什么？",
                        "iPhone 16 Pro 支持 7 天无理由退货，需保持商品完好、配件齐全、包装完整。退货运费由买家承担，质量问题由卖家承担运费。",
                        List.of("chunk_12", "chunk_13"),
                        "knowledge"
                ),
                new EvalCase(
                        "AirPods Pro 的保修期是多久？",
                        "AirPods Pro 保修期为 1 年，自购买之日起计算。保修范围包括硬件故障和制造缺陷，不包括人为损坏和进水。",
                        List.of("chunk_21"),
                        "knowledge"
                ),
                new EvalCase(
                        "退货运费谁承担？",
                        "正常退货运费由买家承担。如果是商品质量问题导致的退货，运费由卖家承担。",
                        List.of("chunk_13"),
                        "knowledge"
                ),
                new EvalCase(
                        "跨境商品能退货吗？",
                        "跨境商品支持退货，但需要在签收后 7 天内提出。退货运费由买家承担，且需要自行办理退货物流。部分商品可能不支持退货，以商品详情页说明为准。",
                        List.of("chunk_35", "chunk_36"),
                        "knowledge"
                ),
                new EvalCase(
                        "质量问题怎么换货？",
                        "质量问题换货流程：1. 在订单详情页提交换货申请并上传质量问题照片 2. 等待客服审核（1-2 个工作日）3. 审核通过后寄回商品，运费由卖家承担 4. 收到商品后 3 个工作日内寄出新商品。",
                        List.of("chunk_08", "chunk_09"),
                        "knowledge"
                ),
                new EvalCase(
                        "Apple Watch Ultra 的防水等级是多少？",
                        "抱歉，当前知识库中没有找到 Apple Watch Ultra 防水等级的相关信息。建议您查看商品详情页或联系人工客服获取准确信息。",
                        List.of(),
                        "knowledge"
                )
        );
    }

    // ========== 模拟 RAG 检索和生成 ==========

    /**
     * 模拟 RAG 检索：实际项目中这里调用向量数据库检索
     * 这里用硬编码模拟，重点展示评测流程
     */
    static Map<String, List<String>> simulateRetrieval() {
        Map<String, List<String>> results = new HashMap<>();
        // 模拟每个问题的 Top-3 检索结果
        results.put("iPhone 16 Pro 的退货政策是什么？",
                List.of("chunk_12", "chunk_05", "chunk_33"));
        results.put("AirPods Pro 的保修期是多久？",
                List.of("chunk_21", "chunk_07", "chunk_44"));
        results.put("退货运费谁承担？",
                List.of("chunk_18", "chunk_29", "chunk_55")); // 没命中 chunk_13
        results.put("跨境商品能退货吗？",
                List.of("chunk_35", "chunk_03", "chunk_67")); // 只命中了 chunk_35
        results.put("质量问题怎么换货？",
                List.of("chunk_08", "chunk_15", "chunk_22")); // 命中了 chunk_08
        results.put("Apple Watch Ultra 的防水等级是多少？",
                List.of("chunk_50", "chunk_51", "chunk_52")); // 全不相关
        return results;
    }

    /**
     * 模拟 RAG 生成：实际项目中这里用检索到的 chunk 内容 + Prompt 调用大模型生成
     * 这里用硬编码模拟不同质量的回答，展示评分效果
     */
    static Map<String, String> simulateGeneration() {
        Map<String, String> results = new HashMap<>();
        results.put("iPhone 16 Pro 的退货政策是什么？",
                "iPhone 16 Pro 支持 7 天无理由退货，需要保持商品完好、配件齐全。退货运费由买家承担，质量问题运费由卖家承担。");
        results.put("AirPods Pro 的保修期是多久？",
                "AirPods Pro 的保修期为 1 年，从购买日期开始计算。保修覆盖硬件故障和制造缺陷，人为损坏和进水不在保修范围内。");
        results.put("退货运费谁承担？",
                "一般情况下退货运费由买家自行承担。不过如果是因为商品本身的质量问题需要退货，运费会由卖家来承担。");
        results.put("跨境商品能退货吗？",
                "跨境商品可以退货，需要在签收后 7 天内申请。退货运费由买家承担。需要注意的是，跨境退货支持全球免费上门取件服务。");  // "全球免费上门取件" 是编造的
        results.put("质量问题怎么换货？",
                "质量问题换货步骤：1. 提交换货申请并上传照片 2. 等待审核 1-2 个工作日 3. 寄回商品（运费卖家承担）4. 收到后 3 个工作日寄出新商品。");
        results.put("Apple Watch Ultra 的防水等级是多少？",
                "抱歉，目前没有找到 Apple Watch Ultra 防水等级的相关信息，建议您查看商品详情页或联系人工客服确认。");
        return results;
    }

    // ========== 检索指标计算 ==========

    /**
     * 计算命中率：Top-K 里有没有包含正确 chunk
     */
    static boolean calculateHit(List<String> retrievedIds, List<String> relevantIds) {
        if (relevantIds.isEmpty()) {
            return false;  // 兜底样本没有相关 chunk 标注，不参与命中判断
        }
        for (String id : retrievedIds) {
            if (relevantIds.contains(id)) {
                return true;
            }
        }
        return false;
    }

    /**
     * 计算倒数排名：正确 chunk 排在第几位
     */
    static double calculateReciprocalRank(List<String> retrievedIds, List<String> relevantIds) {
        if (relevantIds.isEmpty()) {
            return 0.0;  // 兜底样本没有相关 chunk 标注，不参与 MRR 计算
        }
        for (int i = 0; i < retrievedIds.size(); i++) {
            if (relevantIds.contains(retrievedIds.get(i))) {
                return 1.0 / (i + 1);
            }
        }
        return 0.0;  // Top-K 里没有正确答案
    }

    // ========== LLM 评分 ==========

    /**
     * 调用大模型进行评分
     */
    static ScoreResult llmScore(String scorePrompt) throws IOException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", JUDGE_MODEL);

        JsonArray messages = new JsonArray();
        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role", "user");
        userMessage.addProperty("content", scorePrompt);
        messages.add(userMessage);

        requestBody.add("messages", messages);
        requestBody.addProperty("temperature", 0.1);
        requestBody.addProperty("max_tokens", 200);

        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(requestBody.toString(),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            String body = response.body().string();
            JsonObject json = JsonParser.parseString(body).getAsJsonObject();
            String content = json.getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString()
                    .trim();

            // 提取 JSON 部分（模型可能输出额外文字）
            int start = content.indexOf("{");
            int end = content.lastIndexOf("}") + 1;
            if (start >= 0 && end > start) {
                content = content.substring(start, end);
            }

            return gson.fromJson(content, ScoreResult.class);
        }
    }

    /**
     * 忠实度评分
     */
    static ScoreResult scoreFaithfulness(String chunks, String answer) throws IOException {
        String prompt = "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否忠实于给定的参考文档内容。\n\n"
                + "评分标准：\n"
                + "- 5 分：回答完全基于参考文档，没有添加任何文档中没有的信息\n"
                + "- 4 分：回答基本基于参考文档，有极少量合理推断但不影响准确性\n"
                + "- 3 分：回答部分基于参考文档，但添加了一些文档中没有的信息\n"
                + "- 2 分：回答包含较多文档中没有的信息，存在明显编造\n"
                + "- 1 分：回答与参考文档内容严重不符或大量编造\n\n"
                + "参考文档内容：\n" + chunks + "\n\n"
                + "模型的回答：\n" + answer + "\n\n"
                + "请按以下 JSON 格式输出评分结果，不要输出其他内容：\n"
                + "{\"score\": <1-5的整数>, \"label\": \"<faithful/partially_faithful/unfaithful>\", "
                + "\"reason\": \"<简要说明评分理由>\"}";
        return llmScore(prompt);
    }

    /**
     * 相关性评分
     */
    static ScoreResult scoreRelevancy(String query, String answer) throws IOException {
        String prompt = "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否回答了用户的问题。\n\n"
                + "评分标准：\n"
                + "- 5 分：直接、完整地回答了用户的问题\n"
                + "- 4 分：回答了用户的问题，但不够完整或包含了多余信息\n"
                + "- 3 分：部分回答了用户的问题，但遗漏了关键信息\n"
                + "- 2 分：回答与用户的问题有关，但没有真正回答问题\n"
                + "- 1 分：回答与用户的问题完全无关\n\n"
                + "用户问题：\n" + query + "\n\n"
                + "模型的回答：\n" + answer + "\n\n"
                + "请按以下 JSON 格式输出评分结果，不要输出其他内容：\n"
                + "{\"score\": <1-5的整数>, \"label\": \"<relevant/partially_relevant/irrelevant>\", "
                + "\"reason\": \"<简要说明评分理由>\"}";
        return llmScore(prompt);
    }

    /**
     * 正确率评分
     */
    static ScoreResult scoreCorrectness(String query, String expectedAnswer, String actualAnswer) throws IOException {
        String prompt = "你是一个专业的 RAG 系统评估员。你的任务是评估模型的回答是否正确。\n\n"
                + "评分标准：\n"
                + "- 5 分：回答与标准答案的含义完全一致\n"
                + "- 4 分：回答与标准答案基本一致，核心信息正确，细节略有差异\n"
                + "- 3 分：回答部分正确，但遗漏或错误了一些重要信息\n"
                + "- 2 分：回答包含正确信息，但主要结论有误\n"
                + "- 1 分：回答与标准答案完全不一致\n\n"
                + "用户问题：\n" + query + "\n\n"
                + "标准答案：\n" + expectedAnswer + "\n\n"
                + "模型的回答：\n" + actualAnswer + "\n\n"
                + "请按以下 JSON 格式输出评分结果，不要输出其他内容：\n"
                + "{\"score\": <1-5的整数>, \"label\": \"<correct/partially_correct/incorrect>\", "
                + "\"reason\": \"<简要说明评分理由>\"}";
        return llmScore(prompt);
    }

    // ========== 评估报告 ==========

    static void printEvalReport(List<EvalResult> results) {
        System.out.println("=" .repeat(70));
        System.out.println("                    RAG 系统评估报告");
        System.out.println("=" .repeat(70));

        // --- 检索指标 ---
        List<EvalResult> retrievalResults = results.stream()
                .filter(r -> !r.evalCase.relevantChunkIds.isEmpty())
                .toList();
        long hitCount = retrievalResults.stream().filter(r -> r.hit).count();
        double hitRate = (double) hitCount / retrievalResults.size();
        double mrr = retrievalResults.stream()
                .mapToDouble(r -> r.reciprocalRank).average().orElse(0);

        System.out.println("\n【检索阶段指标】");
        System.out.printf("  命中率（Hit Rate）：%.1f%%（%d / %d）%n",
                hitRate * 100, hitCount, retrievalResults.size());
        System.out.printf("  MRR（平均倒数排名）：%.3f%n", mrr);

        // --- 生成指标 ---
        double avgFaithfulness = results.stream()
                .filter(r -> r.faithfulness != null)
                .mapToInt(r -> r.faithfulness.score).average().orElse(0);
        double avgRelevancy = results.stream()
                .filter(r -> r.relevancy != null)
                .mapToInt(r -> r.relevancy.score).average().orElse(0);
        long hallucinationCount = results.stream()
                .filter(r -> r.faithfulness != null && r.faithfulness.score <= 2)
                .count();
        double hallucinationRate = (double) hallucinationCount / results.size();

        System.out.println("\n【生成阶段指标】");
        System.out.printf("  忠实度平均分：%.2f / 5.0%n", avgFaithfulness);
        System.out.printf("  相关性平均分：%.2f / 5.0%n", avgRelevancy);
        System.out.printf("  明显幻觉率：%.1f%%（%d / %d 条存在明显幻觉）%n",
                hallucinationRate * 100, hallucinationCount, results.size());

        // --- 端到端指标 ---
        double avgCorrectness = results.stream()
                .filter(r -> r.correctness != null)
                .mapToInt(r -> r.correctness.score).average().orElse(0);
        long correctCount = results.stream()
                .filter(r -> r.correctness != null && r.correctness.score >= 4)
                .count();
        double correctRate = (double) correctCount / results.size();

        // 兜底率：回答中包含"抱歉""找不到""没有找到"等关键词的比例
        long fallbackCount = results.stream()
                .filter(r -> r.actualAnswer.contains("抱歉") || r.actualAnswer.contains("找不到")
                        || r.actualAnswer.contains("没有找到"))
                .count();
        double fallbackRate = (double) fallbackCount / results.size();

        System.out.println("\n【端到端指标】");
        System.out.printf("  正确率评分均值：%.2f / 5.0%n", avgCorrectness);
        System.out.printf("  答案正确率（≥4 分）：%.1f%%（%d / %d）%n",
                correctRate * 100, correctCount, results.size());
        System.out.printf("  兜底率：%.1f%%（%d / %d）%n",
                fallbackRate * 100, fallbackCount, results.size());

        // --- Bad Case 列表 ---
        System.out.println("\n【Bad Case 列表】（正确率评分 < 4 分的问题）");
        System.out.println("-".repeat(70));
        boolean hasBadCase = false;
        for (EvalResult r : results) {
            if (r.correctness != null && r.correctness.score < 4) {
                hasBadCase = true;
                System.out.printf("  问题：%s%n", r.evalCase.query);
                System.out.printf("  期望答案：%s%n", r.evalCase.expectedAnswer);
                System.out.printf("  实际答案：%s%n", r.actualAnswer);
                System.out.printf("  检索命中：%s | 忠实度：%d 分 | 相关性：%d 分 | 正确率：%d 分%n",
                        r.hit ? "是" : "否",
                        r.faithfulness != null ? r.faithfulness.score : 0,
                        r.relevancy != null ? r.relevancy.score : 0,
                        r.correctness.score);
                // 问题归因
                if (!r.hit) {
                    System.out.println("  → 问题归因：【检索阶段】未命中正确 chunk");
                } else if (r.faithfulness != null && r.faithfulness.score <= 3) {
                    System.out.println("  → 问题归因：【生成阶段】回答与 chunk 内容不够一致，存在编造或额外推断");
                } else {
                    System.out.println("  → 问题归因：【知识库】chunk 内容可能不完整或过时");
                }
                System.out.println("-".repeat(70));
            }
        }
        if (!hasBadCase) {
            System.out.println("  无 Bad Case，所有评测问题的正确率评分均 ≥ 4 分");
        }

        System.out.println("\n" + "=".repeat(70));
    }

    public static void main(String[] args) throws Exception {
        // 1. 构建评测数据集
        List<EvalCase> evalDataset = buildEvalDataset();
        System.out.println("评测数据集：" + evalDataset.size() + " 条");

        // 2. 模拟检索和生成（实际项目中替换为真实的 RAG 流程）
        Map<String, List<String>> retrievalResults = simulateRetrieval();
        Map<String, String> generationResults = simulateGeneration();

        // 模拟 chunk 内容（实际项目中从向量数据库获取）
        Map<String, String> chunkContents = Map.of(
                "chunk_12", "iPhone 16 Pro 支持 7 天无理由退货，需保持商品完好、配件齐全、包装完整。",
                "chunk_13", "退货运费由买家承担，质量问题由卖家承担运费。",
                "chunk_21", "AirPods Pro 保修期为 1 年，自购买之日起计算。保修范围包括硬件故障和制造缺陷，不包括人为损坏和进水。",
                "chunk_35", "跨境商品支持退货，需在签收后 7 天内提出。退货运费由买家承担，需自行办理退货物流。",
                "chunk_08", "质量问题换货流程：1. 提交换货申请并上传照片 2. 等待审核 1-2 个工作日 3. 寄回商品运费由卖家承担 4. 收到后 3 个工作日寄出新商品。"
        );

        // 3. 逐条评测
        List<EvalResult> evalResults = new ArrayList<>();
        for (int i = 0; i < evalDataset.size(); i++) {
            EvalCase evalCase = evalDataset.get(i);
            System.out.printf("\n评测第 %d/%d 条：%s%n", i + 1, evalDataset.size(), evalCase.query);

            EvalResult result = new EvalResult();
            result.evalCase = evalCase;

            // 获取模拟的检索和生成结果
            result.retrievedChunkIds = retrievalResults.getOrDefault(evalCase.query, List.of());
            result.actualAnswer = generationResults.getOrDefault(evalCase.query, "");

            // 计算检索指标
            result.hit = calculateHit(result.retrievedChunkIds, evalCase.relevantChunkIds);
            result.reciprocalRank = calculateReciprocalRank(result.retrievedChunkIds, evalCase.relevantChunkIds);
            if (evalCase.relevantChunkIds.isEmpty()) {
                System.out.println("  检索评估：跳过（兜底样本，无相关 chunk 标注）");
            } else {
                System.out.printf("  检索命中：%s，倒数排名：%.2f%n", result.hit ? "是" : "否", result.reciprocalRank);
            }

            // 组装检索到的 chunk 内容
            StringBuilder chunkText = new StringBuilder();
            for (String chunkId : result.retrievedChunkIds) {
                if (chunkContents.containsKey(chunkId)) {
                    chunkText.append("[").append(chunkId).append("] ")
                            .append(chunkContents.get(chunkId)).append("\n");
                }
            }
            String chunks = !chunkText.isEmpty() ? chunkText.toString() : "（未检索到相关内容）";

            // LLM 评分（三个维度）
            System.out.println("  正在评分...");
            result.faithfulness = scoreFaithfulness(chunks, result.actualAnswer);
            System.out.printf("  忠实度：%d 分 - %s%n", result.faithfulness.score, result.faithfulness.reason);

            result.relevancy = scoreRelevancy(evalCase.query, result.actualAnswer);
            System.out.printf("  相关性：%d 分 - %s%n", result.relevancy.score, result.relevancy.reason);

            result.correctness = scoreCorrectness(evalCase.query, evalCase.expectedAnswer, result.actualAnswer);
            System.out.printf("  正确率：%d 分 - %s%n", result.correctness.score, result.correctness.reason);

            evalResults.add(result);
        }

        // 4. 输出评估报告
        System.out.println();
        printEvalReport(evalResults);
    }
}
