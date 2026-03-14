package com.nageoffer.ai.tinyrag.service.rag;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.service.RerankService;
import com.nageoffer.ai.tinyrag.service.RerankService.RerankItem;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

import lombok.extern.slf4j.Slf4j;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;
import org.springframework.stereotype.Component;

@Slf4j
@Component
public class RerankDocumentPostProcessor implements DocumentPostProcessor {

    private final RerankService rerankService;
    private final RAGProperties ragProperties;

    public RerankDocumentPostProcessor(RerankService rerankService, RAGProperties ragProperties) {
        this.rerankService = rerankService;
        this.ragProperties = ragProperties;
    }

    @Override
    public @NonNull List<Document> process(@NonNull Query query, @NonNull List<Document> documents) {
        if (documents.isEmpty()) {
            return List.of();
        }

        int safeTopN = Math.max(1, ragProperties.getRerankTopN());
        List<Document> validCandidates = new ArrayList<>();
        List<String> candidateTexts = new ArrayList<>();
        for (Document document : documents) {
            String text = safeText(document);
            if (text.isBlank()) {
                continue;
            }
            validCandidates.add(document);
            candidateTexts.add(truncate(text, ragProperties.getRerankMaxDocumentChars()));
        }

        if (validCandidates.isEmpty()) {
            return List.of();
        }

        try {
            List<RerankItem> rerankResults = rerankService.rerank(query.text(), candidateTexts, safeTopN);
            List<Document> reranked = pickByRerankResults(validCandidates, rerankResults, safeTopN);
            if (!reranked.isEmpty()) {
                return reranked;
            }
        } catch (Exception ex) {
            log.warn("Rerank failed, fallback to vector score. reason={}", ex.getMessage());
        }

        return fallbackByVectorScore(validCandidates, safeTopN);
    }

    private List<Document> pickByRerankResults(List<Document> candidates, List<RerankItem> results, int topN) {
        if (results == null || results.isEmpty()) {
            return List.of();
        }

        List<Document> picked = new ArrayList<>();
        Set<Integer> seen = new LinkedHashSet<>();
        for (RerankItem result : results) {
            int index = result.index();
            if (index < 0 || index >= candidates.size()) {
                continue;
            }
            if (!seen.add(index)) {
                continue;
            }
            picked.add(candidates.get(index));
            if (picked.size() >= topN) {
                break;
            }
        }
        return picked;
    }

    private List<Document> fallbackByVectorScore(List<Document> candidates, int topN) {
        return candidates.stream()
                .sorted(Comparator.comparingDouble(this::vectorScore).reversed())
                .limit(topN)
                .toList();
    }

    private double vectorScore(Document document) {
        if (document == null || document.getScore() == null) {
            return 0.0;
        }
        return document.getScore();
    }

    private String safeText(Document document) {
        if (document == null || document.getText() == null) {
            return "";
        }
        return document.getText().trim();
    }

    private String truncate(String value, int maxLen) {
        if (value == null || value.length() <= maxLen) {
            return value;
        }
        return value.substring(0, maxLen);
    }
}
