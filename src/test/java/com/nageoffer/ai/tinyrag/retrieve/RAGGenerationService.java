package com.nageoffer.ai.tinyrag.retrieve;

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
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class RAGGenerationService {

    // SiliconFlow API 配置
    private static final String API_URL = "https://api.siliconflow.cn/v1/chat/completions";
    private static final String API_KEY = "sk-xlvfsxatnazrvtfqaklospaokrrohsuewwxgwrqvyxifpbpp";
    private static final String MODEL = "Qwen/Qwen2.5-7B-Instruct";

    private static final OkHttpClient client = new OkHttpClient.Builder()
            .connectTimeout(30, TimeUnit.SECONDS)
            .readTimeout(60, TimeUnit.SECONDS)
            .build();

    private static final Gson gson = new Gson();

    /**
     * System Prompt 模板——生产级版本
     */
    private static final String SYSTEM_PROMPT = """
            你是一名专业的电商客服助手。你的任务是根据【参考资料】中的信息，准确回答用户的问题。

            【角色与边界】
            - 你只负责回答与商品售后、退换货、物流配送相关的问题。
            - 如果用户的问题超出这个范围，请礼貌地告知用户，并引导回售后相关话题。
            - 不要回答涉及品牌对比、价格预测、个人观点等主观性问题。

            【回答规则】
            1. 只基于【参考资料】中的内容回答问题，不要使用你自己的知识。
            2. 如果【参考资料】中没有足够的信息，请明确回答："根据现有资料，暂时无法回答该问题。建议您联系人工客服获取更多帮助。"
            3. 不要编造任何【参考资料】中没有提到的信息，包括数字、日期、金额等。
            4. 如果多条参考资料的信息存在冲突，请指出冲突并告知用户以最新的资料为准。

            【引用规则】
            - 回答时请引用参考资料的编号，格式为 [1]、[2] 等，标注在相关句子的末尾。
            - 如果一句话的信息来自多条参考资料，请同时标注多个编号，如 [1][3]。
            - 只引用你实际使用到的参考资料。

            【格式要求】
            - 用简洁、友好的语气回答。
            - 回答要覆盖用户问题的核心要点，补充必要的注意事项，但不要展开无关的背景知识。
            """;

    /**
     * 步骤一：把检索到的 chunk 列表组装成带编号的上下文
     */
    public String buildContext(List<RetrievedChunk> chunks) {
        StringBuilder context = new StringBuilder("【参考资料】\n\n");
        for (int i = 0; i < chunks.size(); i++) {
            RetrievedChunk chunk = chunks.get(i);
            context.append(String.format("[%d] 来源：%s | 更新时间：%s\n",
                    i + 1, chunk.getSource(), chunk.getUpdateTime()));
            context.append(chunk.getContent()).append("\n\n");
        }
        return context.toString();
    }

    /**
     * 步骤二：调用大模型 Chat API
     */
    public String callLlm(String systemPrompt, String context, String userQuery)
            throws IOException {

        // 拼接完整的用户消息：上下文 + 用户问题
        String userMessage = context + "【用户问题】\n" + userQuery;

        // 构建请求体
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", MODEL);
        requestBody.addProperty("temperature", 0.1);  // 低 Temperature，减少随机性
        requestBody.addProperty("max_tokens", 1024);

        JsonArray messages = new JsonArray();

        // system 消息
        JsonObject systemMsg = new JsonObject();
        systemMsg.addProperty("role", "system");
        systemMsg.addProperty("content", systemPrompt);
        messages.add(systemMsg);

        // user 消息
        JsonObject userMsg = new JsonObject();
        userMsg.addProperty("role", "user");
        userMsg.addProperty("content", userMessage);
        messages.add(userMsg);

        requestBody.add("messages", messages);

        // 发送请求
        Request request = new Request.Builder()
                .url(API_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(
                        gson.toJson(requestBody),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = client.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new IOException("API 调用失败，状态码：" + response.code()
                        + "，响应：" + response.body().string());
            }
            JsonObject responseJson = gson.fromJson(response.body().string(), JsonObject.class);
            return responseJson
                    .getAsJsonArray("choices")
                    .get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        }
    }

    /**
     * 步骤三：解析模型回答中的引用编号
     */
    public List<RAGResponse.CitationInfo> parseCitations(
            String answer, List<RetrievedChunk> chunks) {

        Set<Integer> citedIndexes = new TreeSet<>();
        Pattern pattern = Pattern.compile("\\[(\\d+)]");
        Matcher matcher = pattern.matcher(answer);
        while (matcher.find()) {
            citedIndexes.add(Integer.parseInt(matcher.group(1)));
        }

        List<RAGResponse.CitationInfo> citations = new ArrayList<>();
        for (int index : citedIndexes) {
            if (index >= 1 && index <= chunks.size()) {
                RetrievedChunk chunk = chunks.get(index - 1);
                RAGResponse.CitationInfo info = new RAGResponse.CitationInfo();
                info.setIndex(index);
                info.setSource(chunk.getSource());
                info.setSourceUrl(chunk.getSourceUrl());
                info.setChunkContent(chunk.getContent());
                citations.add(info);
            }
        }
        return citations;
    }

    /**
     * 完整的 RAG 生成流程：检索 → 组装 → 生成 → 解析
     *
     * @param chunks    经过混合检索 + 重排序后的 Top-K chunk
     * @param userQuery 用户的原始问题
     * @return 包含回答和引用信息的 RAGResponse
     */
    public RAGResponse generate(List<RetrievedChunk> chunks, String userQuery)
            throws IOException {

        // 1. 组装上下文
        String context = buildContext(chunks);

        // 2. 调用大模型
        String answer = callLlm(SYSTEM_PROMPT, context, userQuery);

        // 3. 解析引用
        List<RAGResponse.CitationInfo> citations = parseCitations(answer, chunks);

        // 4. 组装结果
        RAGResponse response = new RAGResponse();
        response.setAnswer(answer);
        response.setCitations(citations);
        return response;
    }
}