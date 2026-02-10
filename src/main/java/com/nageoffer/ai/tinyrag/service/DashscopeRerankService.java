package com.nageoffer.ai.tinyrag.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.nageoffer.ai.tinyrag.config.RAGProperties;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;

@Service
public class DashscopeRerankService {

    private final RAGProperties ragProperties;
    private final RestClient restClient;
    private final String apiKey;

    public DashscopeRerankService(RAGProperties ragProperties,
                                  @Value("${spring.ai.openai.api-key:}") String apiKey) {
        this.ragProperties = ragProperties;
        this.apiKey = apiKey;
        this.restClient = RestClient.builder().build();
    }

    public List<RerankItem> rerank(String query, List<String> documents, int topN) {
        if (!StringUtils.hasText(query) || documents == null || documents.isEmpty()) {
            return List.of();
        }
        if (!StringUtils.hasText(apiKey) || "your-api-key".equalsIgnoreCase(apiKey.trim())) {
            throw new IllegalStateException("DashScope API Key 未配置，无法执行 Rerank");
        }

        String model = ragProperties.getRerankModel();
        Map<String, Object> primaryRequest = buildRequest(query, documents, topN);
        JsonNode body;
        try {
            body = callRerankApi(primaryRequest);
        } catch (RuntimeException ex) {
            if (!isQwen3RerankModel(model)) {
                throw ex;
            }
            body = callRerankApi(buildLegacyRequest(query, documents, topN, model));
        }

        return parseResults(body);
    }

    private Map<String, Object> buildRequest(String query, List<String> documents, int topN) {
        int safeTopN = Math.max(1, Math.min(topN, documents.size()));
        String model = ragProperties.getRerankModel();

        if (isQwen3RerankModel(model)) {
            Map<String, Object> request = new LinkedHashMap<>();
            request.put("model", model);
            request.put("query", query);
            request.put("documents", documents);
            request.put("top_n", safeTopN);
            return request;
        }

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("model", model);
        request.put("input", Map.of("query", query, "documents", documents));
        request.put("parameters", Map.of("return_documents", true, "top_n", safeTopN));
        return request;
    }

    private Map<String, Object> buildLegacyRequest(String query, List<String> documents, int topN, String model) {
        int safeTopN = Math.max(1, Math.min(topN, documents.size()));
        return Map.of(
                "model", model,
                "input", Map.of("query", query, "documents", documents),
                "parameters", Map.of("return_documents", true, "top_n", safeTopN)
        );
    }

    private JsonNode callRerankApi(Map<String, Object> requestBody) {
        return restClient.post()
                .uri(ragProperties.getRerankEndpoint())
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                .contentType(MediaType.APPLICATION_JSON)
                .body(requestBody)
                .retrieve()
                .body(JsonNode.class);
    }

    private boolean isQwen3RerankModel(String model) {
        return model != null && model.toLowerCase(Locale.ROOT).startsWith("qwen3-rerank");
    }

    private List<RerankItem> parseResults(JsonNode root) {
        if (root == null) {
            return List.of();
        }

        JsonNode results = root.path("output").path("results");
        if (!results.isArray()) {
            results = root.path("results");
        }
        if (!results.isArray()) {
            return List.of();
        }

        List<RerankItem> parsed = new ArrayList<>();
        for (JsonNode item : results) {
            int index = item.path("index").asInt(-1);
            if (index < 0) {
                continue;
            }
            double score = extractScore(item);
            parsed.add(new RerankItem(index, score));
        }
        return parsed;
    }

    private double extractScore(JsonNode item) {
        if (item == null) {
            return 0.0;
        }
        if (item.has("relevance_score")) {
            return item.path("relevance_score").asDouble(0.0);
        }
        if (item.has("score")) {
            return item.path("score").asDouble(0.0);
        }
        return 0.0;
    }

    public record RerankItem(int index, double score) {
    }
}
