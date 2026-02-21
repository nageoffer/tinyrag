package com.nageoffer.ai.tinyrag.milvus.rerank;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class SiliconFlowRerankDemo {

    private static final String API_KEY = "你的 SiliconFlow API Key";
    private static final String RERANK_URL = "https://api.siliconflow.cn/v1/rerank";
    private static final String MODEL = "BAAI/bge-reranker-v2-m3";
    private static final Gson GSON = new Gson();
    private static final OkHttpClient HTTP = new OkHttpClient();

    public static class RerankItem {
        public int index;
        public double score;
        public String text;
    }

    public static List<RerankItem> rerank(String query, List<String> candidates, int topN) throws IOException {
        JsonObject body = new JsonObject();
        body.addProperty("model", MODEL);
        body.addProperty("query", query);
        body.add("documents", GSON.toJsonTree(candidates));
        body.addProperty("top_n", topN);
        body.addProperty("return_documents", true);

        Request request = new Request.Builder()
                .url(RERANK_URL)
                .addHeader("Authorization", "Bearer " + API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(GSON.toJson(body), MediaType.parse("application/json")))
                .build();

        try (Response response = HTTP.newCall(request).execute()) {
            if (!response.isSuccessful()) {
                throw new RuntimeException("rerank 调用失败，HTTP=" + response.code());
            }

            JsonObject resp = GSON.fromJson(response.body().string(), JsonObject.class);
            JsonArray results = resp.getAsJsonArray("results");

            List<RerankItem> items = new ArrayList<>();
            for (int i = 0; i < results.size(); i++) {
                JsonObject one = results.get(i).getAsJsonObject();
                RerankItem item = new RerankItem();
                item.index = one.get("index").getAsInt();
                item.score = one.get("relevance_score").getAsDouble();

                if (one.has("document") && one.get("document").isJsonObject()) {
                    JsonObject doc = one.getAsJsonObject("document");
                    item.text = doc.has("text") ? doc.get("text").getAsString() : candidates.get(item.index);
                } else {
                    item.text = candidates.get(item.index);
                }

                items.add(item);
            }

            items.sort(Comparator.comparingDouble((RerankItem x) -> x.score).reversed());
            return items;
        }
    }

    public static void main(String[] args) throws IOException {
        String query = "订单号 2026012345 的物流状态";
        List<String> candidates = List.of(
                "物流配送时效说明：全国大部分地区 48 小时内发货",
                "订单号 2026012345：已于 2026-02-18 14:21 从杭州仓发出，承运商顺丰，当前状态运输中",
                "如何查询订单物流：登录账号后进入订单详情页，点击物流跟踪",
                "物流异常处理流程：如遇物流异常，请联系客服处理",
                "快递公司合作列表：顺丰、圆通、中通、韵达"
        );

        List<RerankItem> results = rerank(query, candidates, 3);

        System.out.println("重排序后的 Top-3：");
        for (int i = 0; i < results.size(); i++) {
            RerankItem item = results.get(i);
            System.out.println("Top-" + (i + 1) + " score=" + item.score);
            System.out.println("  " + item.text);
        }
    }
}