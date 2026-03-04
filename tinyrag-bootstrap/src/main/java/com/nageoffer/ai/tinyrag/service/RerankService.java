package com.nageoffer.ai.tinyrag.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.nageoffer.ai.tinyrag.config.RAGProperties;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;

@Service
public class RerankService {

    private final RAGProperties ragProperties;
    private final RestClient restClient;
    private final ObjectMapper objectMapper = new ObjectMapper();
    private final String apiKey;

    public RerankService(RAGProperties ragProperties,
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
            throw new IllegalStateException("Rerank API Key 未配置，无法执行 rerank");
        }
        if (!StringUtils.hasText(ragProperties.getRerankEndpoint())) {
            throw new IllegalStateException("Rerank endpoint 未配置，无法执行 rerank");
        }

        JsonNode body = callRerankApi(buildRequest(query, documents, topN));
        return parseResults(body);
    }

    private Map<String, Object> buildRequest(String query, List<String> documents, int topN) {
        int safeTopN = Math.max(1, Math.min(topN, documents.size()));

        Map<String, Object> request = new LinkedHashMap<>();
        request.put("model", ragProperties.getRerankModel());
        request.put("query", query);
        request.put("documents", documents);
        request.put("top_n", safeTopN);
        return request;
    }

    private JsonNode callRerankApi(Map<String, Object> requestBody) {
        String responseBody = restClient.post()
                .uri(ragProperties.getRerankEndpoint())
                .header(HttpHeaders.AUTHORIZATION, "Bearer " + apiKey)
                .contentType(MediaType.APPLICATION_JSON)
                .body(requestBody)
                .retrieve()
                .body(String.class);
        if (!StringUtils.hasText(responseBody)) {
            return null;
        }
        try {
            return objectMapper.readTree(responseBody);
        } catch (JsonProcessingException ex) {
            throw new IllegalStateException("Rerank response parse failed", ex);
        }
    }

    private List<RerankItem> parseResults(JsonNode root) {
        if (root == null) {
            return List.of();
        }

        JsonNode results = root.path("results");
        if (!results.isArray()) {
            results = root.path("output").path("results");
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
