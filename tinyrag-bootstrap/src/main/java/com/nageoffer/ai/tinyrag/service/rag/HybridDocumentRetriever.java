package com.nageoffer.ai.tinyrag.service.rag;

import com.nageoffer.ai.tinyrag.config.RAGProperties;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;

@Slf4j
@RequiredArgsConstructor
public class HybridDocumentRetriever implements DocumentRetriever {

    private final VectorStoreDocumentRetriever vectorRetriever;
    private final KeywordDocumentRetriever keywordRetriever;
    private final RAGProperties ragProperties;

    @Override
    public @NonNull List<Document> retrieve(@NonNull Query query) {
        CompletableFuture<List<Document>> vectorFuture = CompletableFuture.supplyAsync(() -> vectorRetriever.retrieve(query));
        CompletableFuture<List<Document>> keywordFuture = CompletableFuture.supplyAsync(() -> keywordRetriever.retrieve(query));

        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);

        List<Document> vectorDocs = getQuietly(vectorFuture, deadline, "向量");
        List<Document> keywordDocs = getQuietly(keywordFuture, deadline, "关键词");

        if (vectorDocs.isEmpty() && keywordDocs.isEmpty()) {
            return List.of();
        }
        if (keywordDocs.isEmpty()) {
            log.info("[Hybrid] 仅向量检索返回 {} 个文档", vectorDocs.size());
            return vectorDocs;
        }
        if (vectorDocs.isEmpty()) {
            log.info("[Hybrid] 仅关键词检索返回 {} 个文档", keywordDocs.size());
            return keywordDocs;
        }

        List<Document> fused = rrfFusion(vectorDocs, keywordDocs);
        log.info("[Hybrid] RRF 融合完成: 向量={}, 关键词={}, 融合后={}",
                vectorDocs.size(), keywordDocs.size(), fused.size());
        return fused;
    }

    private List<Document> getQuietly(CompletableFuture<List<Document>> future,
                                      long deadlineNanos, String label) {
        try {
            long remaining = deadlineNanos - System.nanoTime();
            if (remaining <= 0) {
                future.cancel(true);
                throw new TimeoutException("deadline exceeded");
            }
            return future.get(remaining, TimeUnit.NANOSECONDS);
        } catch (Exception e) {
            log.warn("[Hybrid] {}检索失败, 降级: {}", label, e.getMessage());
            return List.of();
        }
    }

    private List<Document> rrfFusion(List<Document> vectorDocs, List<Document> keywordDocs) {
        int k = ragProperties.getRrfK();
        int topK = ragProperties.getRetrieveTopK();

        Map<String, Double> scoreMap = new HashMap<>();
        Map<String, Document> docMap = new LinkedHashMap<>();

        accumulateScores(vectorDocs, k, scoreMap, docMap);
        accumulateScores(keywordDocs, k, scoreMap, docMap);

        return scoreMap.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .limit(topK)
                .map(entry -> docMap.get(entry.getKey()))
                .filter(Objects::nonNull)
                .toList();
    }

    private void accumulateScores(List<Document> docs, int k,
                                  Map<String, Double> scoreMap,
                                  Map<String, Document> docMap) {
        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i);
            String docId = doc.getId();
            scoreMap.merge(docId, 1.0 / (k + i + 1), Double::sum);
            docMap.putIfAbsent(docId, doc);
        }
    }
}
