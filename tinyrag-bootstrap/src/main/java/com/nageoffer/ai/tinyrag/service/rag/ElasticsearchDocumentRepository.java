package com.nageoffer.ai.tinyrag.service.rag;

import cn.hutool.core.collection.CollUtil;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.nageoffer.ai.tinyrag.config.RAGProperties;

import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.document.Document;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.util.StringUtils;
import org.springframework.web.client.RestClient;

@Slf4j
@Component
public class ElasticsearchDocumentRepository {

    private static final String INDEX_NAME = "tinyrag_chunks";
    private static final Type MAP_TYPE = new TypeToken<Map<String, Object>>() {
    }.getType();

    private final RAGProperties ragProperties;
    private final Gson gson = new Gson();
    private RestClient restClient;

    public ElasticsearchDocumentRepository(RAGProperties ragProperties) {
        this.ragProperties = ragProperties;
    }

    @PostConstruct
    public void init() {
        this.restClient = RestClient.builder()
                .baseUrl(ragProperties.getEsUrl())
                .build();
        initIndex();
    }

    private void initIndex() {
        try {
            String analyzer = ragProperties.getEsAnalyzer();

            Map<String, Object> body = getStringObjectMap(analyzer);

            restClient.put()
                    .uri("/{index}", INDEX_NAME)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(gson.toJson(body))
                    .retrieve()
                    .toBodilessEntity();

            log.info("[ES] 索引 {} 创建成功, analyzer={}", INDEX_NAME, analyzer);
        } catch (Exception e) {
            String msg = e.getMessage();
            if (msg != null && msg.contains("resource_already_exists_exception")) {
                log.info("[ES] 索引 {} 已存在, 跳过创建", INDEX_NAME);
            } else {
                log.warn("[ES] 索引初始化失败, 关键词检索不可用: {}", msg);
            }
        }
    }

    private static @NonNull Map<String, Object> getStringObjectMap(String analyzer) {
        Map<String, Object> contentField = Map.of("type", "text", "analyzer", analyzer);
        Map<String, Object> keywordField = Map.of("type", "keyword");
        Map<String, Object> integerField = Map.of("type", "integer");

        Map<String, Object> properties = Map.of(
                "content", contentField,
                "source", keywordField,
                "kb", keywordField,
                "file_type", keywordField,
                "chunk_index", integerField,
                "doc_id", keywordField
        );

        return Map.of("mappings", Map.of("properties", properties));
    }

    public void indexDocuments(List<Document> documents) {
        if (CollUtil.isEmpty(documents)) {
            return;
        }
        try {
            StringBuilder ndjson = new StringBuilder();
            for (Document doc : documents) {
                Map<String, Object> action = Map.of("index", Map.of("_index", INDEX_NAME, "_id", doc.getId()));
                ndjson.append(gson.toJson(action)).append("\n");

                Map<String, Object> source = new HashMap<>();
                source.put("content", doc.getText());
                source.put("source", doc.getMetadata().getOrDefault("source", ""));
                source.put("kb", doc.getMetadata().getOrDefault("kb", ""));
                source.put("file_type", doc.getMetadata().getOrDefault("file_type", ""));
                source.put("chunk_index", doc.getMetadata().getOrDefault("chunk_index", 0));
                source.put("doc_id", doc.getId());
                ndjson.append(gson.toJson(source)).append("\n");
            }

            String response = restClient.post()
                    .uri("/_bulk")
                    .contentType(MediaType.parseMediaType("application/x-ndjson;charset=UTF-8"))
                    .body(ndjson.toString())
                    .retrieve()
                    .body(String.class);

            Map<String, Object> result = gson.fromJson(response, MAP_TYPE);
            if (Boolean.TRUE.equals(result.get("errors"))) {
                @SuppressWarnings("unchecked")
                List<Map<String, Object>> items = (List<Map<String, Object>>) result.get("items");
                if (items != null) {
                    for (Map<String, Object> item : items) {
                        @SuppressWarnings("unchecked")
                        Map<String, Object> action = (Map<String, Object>) item.values().iterator().next();
                        @SuppressWarnings("unchecked")
                        Map<String, Object> error = (Map<String, Object>) action.get("error");
                        if (error != null) {
                            log.warn("[ES] 文档写入失败: id={}, type={}, reason={}",
                                    action.get("_id"), error.get("type"), error.get("reason"));
                        }
                    }
                }
            }
            log.info("[ES] 批量写入完成, 共 {} 个文档", documents.size());
        } catch (Exception e) {
            log.warn("[ES] 文档写入失败: {}", e.getMessage());
        }
    }

    @SuppressWarnings("unchecked")
    public List<Document> search(String queryText, String kb, int topK) {
        try {
            Map<String, Object> matchQuery = Map.of("match", Map.of("content", queryText));

            Map<String, Object> query;
            if (StringUtils.hasText(kb)) {
                Map<String, Object> termFilter = Map.of("term", Map.of("kb", kb));
                query = Map.of("bool", Map.of("must", matchQuery, "filter", termFilter));
            } else {
                query = matchQuery;
            }

            Map<String, Object> body = Map.of("query", query, "size", topK);

            String response = restClient.post()
                    .uri("/{index}/_search", INDEX_NAME)
                    .contentType(MediaType.APPLICATION_JSON)
                    .body(gson.toJson(body))
                    .retrieve()
                    .body(String.class);

            Map<String, Object> result = gson.fromJson(response, MAP_TYPE);
            assert result != null;
            Map<String, Object> hits = (Map<String, Object>) result.get("hits");
            List<Map<String, Object>> hitList = (List<Map<String, Object>>) hits.get("hits");

            List<Document> results = new ArrayList<>();
            if (hitList == null) {
                return results;
            }

            for (Map<String, Object> hit : hitList) {
                Map<String, Object> source = (Map<String, Object>) hit.get("_source");
                if (source == null) {
                    continue;
                }

                Map<String, Object> metadata = new HashMap<>();
                metadata.put("source", source.getOrDefault("source", ""));
                metadata.put("kb", source.getOrDefault("kb", ""));
                metadata.put("file_type", source.getOrDefault("file_type", ""));
                metadata.put("chunk_index", source.getOrDefault("chunk_index", 0));

                String docId = (String) source.getOrDefault("doc_id", hit.get("_id"));
                double score = hit.get("_score") instanceof Number n ? n.doubleValue() : 0.0;

                results.add(Document.builder()
                        .id(docId)
                        .text((String) source.getOrDefault("content", ""))
                        .metadata(metadata)
                        .score(score)
                        .build());
            }

            log.info("[ES] BM25 检索完成, query='{}', kb='{}', 返回 {} 个文档", queryText, kb, results.size());
            return results;
        } catch (Exception e) {
            log.warn("[ES] BM25 检索失败: {}", e.getMessage());
            return List.of();
        }
    }
}
