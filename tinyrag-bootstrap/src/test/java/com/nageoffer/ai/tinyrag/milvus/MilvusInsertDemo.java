package com.nageoffer.ai.tinyrag.milvus;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.response.InsertResp;
import okhttp3.*;

import java.io.IOException;
import java.util.*;

public class MilvusInsertDemo {

    private static final String SILICONFLOW_API_KEY = "你的 SiliconFlow API Key";
    private static final String EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings";
    private static final String EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B";
    private static final Gson GSON = new Gson();
    private static final OkHttpClient HTTP_CLIENT = new OkHttpClient();

    public static void main(String[] args) throws IOException {
        // 连接 Milvus（省略，同上一节）
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build();
        MilvusClientV2 client = new MilvusClientV2(connectConfig);

        // 模拟电商客服知识库的 chunk 数据
        List<String> chunkTexts = List.of(
                "退货政策：自签收之日起 7 天内，商品未拆封、不影响二次销售的情况下，支持无理由退货。退货运费由买家承担，质量问题除外。",
                "退货政策：生鲜食品、定制商品、贴身衣物等特殊商品不支持无理由退货。如有质量问题，请在签收后 48 小时内联系客服并提供照片凭证。",
                "物流规则：普通商品下单后 48 小时内发货，预售商品以商品详情页标注的发货时间为准。偏远地区（新疆、西藏、青海等）可能需要额外 2~3 天。",
                "物流规则：支持顺丰、中通、圆通、韵达等主流快递。默认使用中通快递，如需指定快递公司，请在下单时备注，可能产生额外运费。",
                "促销活动：2026 年春节大促，全场满 300 减 50，满 500 减 100。活动时间：2026 年 1 月 20 日至 2 月 5 日。优惠券不可叠加使用。"
        );
        List<String> docIds = List.of("doc_return_001", "doc_return_001", "doc_logistics_001", "doc_logistics_001", "doc_promo_001");
        List<String> categories = List.of("return_policy", "return_policy", "logistics", "logistics", "promotion");

        // 调用 Embedding API 生成向量
        List<List<Float>> vectors = getEmbeddings(chunkTexts);

        // 组装插入数据
        List<JsonObject> rows = new ArrayList<>();
        for (int i = 0; i < chunkTexts.size(); i++) {
            JsonObject row = new JsonObject();
            row.addProperty("chunk_text", chunkTexts.get(i));
            row.addProperty("doc_id", docIds.get(i));
            row.addProperty("category", categories.get(i));
            row.add("vector", GSON.toJsonTree(vectors.get(i)));
            rows.add(row);
        }

        // 插入 Milvus
        InsertReq insertReq = InsertReq.builder()
                .collectionName("customer_service_chunks")
                .data(rows)
                .build();
        InsertResp insertResp = client.insert(insertReq);
        System.out.println("插入成功，数量：" + insertResp.getInsertCnt());
    }

    /**
     * 调用 SiliconFlow Embedding API，批量生成向量
     */
    private static List<List<Float>> getEmbeddings(List<String> texts) throws IOException {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model", EMBEDDING_MODEL);
        requestBody.add("input", GSON.toJsonTree(texts));

        Request request = new Request.Builder()
                .url(EMBEDDING_URL)
                .addHeader("Authorization", "Bearer " + SILICONFLOW_API_KEY)
                .addHeader("Content-Type", "application/json")
                .post(RequestBody.create(GSON.toJson(requestBody),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = HTTP_CLIENT.newCall(request).execute()) {
            String body = Objects.requireNonNull(response.body()).string();
            JsonObject json = GSON.fromJson(body, JsonObject.class);
            JsonArray dataArray = json.getAsJsonArray("data");

            List<List<Float>> vectors = new ArrayList<>();
            for (int i = 0; i < dataArray.size(); i++) {
                JsonArray embeddingArray = dataArray.get(i).getAsJsonObject()
                        .getAsJsonArray("embedding");
                List<Float> vector = new ArrayList<>();
                for (int j = 0; j < embeddingArray.size(); j++) {
                    vector.add(embeddingArray.get(j).getAsFloat());
                }
                vectors.add(vector);
            }
            return vectors;
        }
    }
}
