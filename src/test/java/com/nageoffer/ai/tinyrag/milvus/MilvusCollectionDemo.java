package com.nageoffer.ai.tinyrag.milvus;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.BaseVector;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public class MilvusCollectionDemo {

    private static final String SILICONFLOW_API_KEY = "你的 SiliconFlow API Key";
    private static final String EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings";
    private static final String EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B";
    private static final Gson GSON = new Gson();
    private static final OkHttpClient HTTP_CLIENT = new OkHttpClient();

    public static void main(String[] args) throws IOException {
        ConnectConfig connectConfig = ConnectConfig.builder()
                .uri("http://localhost:19530")
                .build();
        MilvusClientV2 client = new MilvusClientV2(connectConfig);

        // 用户的问题
        String query = "买了东西不想要了怎么退货？";

        // 把问题向量化（复用前面的 getEmbeddings 方法）
        List<List<Float>> queryVectors = getEmbeddings(List.of(query));

        List<BaseVector> milvusQueryVectors = queryVectors.stream()
                .map(FloatVec::new)   // FloatVec(List<Float>)
                .collect(java.util.stream.Collectors.toList());

        // 执行向量检索
        SearchReq searchReq = SearchReq.builder()
                .collectionName("customer_service_chunks")
                .data(milvusQueryVectors)           // 查询向量
                .topK(3)                      // 返回最相似的 3 个结果
                .outputFields(List.of("chunk_text", "doc_id", "category"))  // 需要返回的字段
                .annsField("vector")          // 指定在哪个向量字段上检索
                .searchParams(Map.of("ef", 256))  // HNSW 检索时的搜索宽度
                .build();

        SearchResp searchResp = client.search(searchReq);

        // 输出检索结果
        List<List<SearchResp.SearchResult>> results = searchResp.getSearchResults();
        for (List<SearchResp.SearchResult> resultList : results) {
            System.out.println("=== 检索结果 ===");
            for (int i = 0; i < resultList.size(); i++) {
                SearchResp.SearchResult result = resultList.get(i);
                System.out.println("Top-" + (i + 1) + "：");
                System.out.println("  相似度分数：" + result.getScore());
                System.out.println("  分类：" + result.getEntity().get("category"));
                System.out.println("  文档ID：" + result.getEntity().get("doc_id"));
                System.out.println("  内容：" + result.getEntity().get("chunk_text"));
                System.out.println();
            }
        }
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
