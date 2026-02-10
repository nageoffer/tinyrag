package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.service.DashscopeRerankService.RerankItem;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;

@Service
public class RAGService {

    private static final Logger log = LoggerFactory.getLogger(RAGService.class);

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final RAGProperties ragProperties;
    private final DashscopeRerankService dashscopeRerankService;
    private final Resource rewriteSystemPrompt;
    private final Resource rewriteUserPrompt;
    private final Resource answerSystemPrompt;
    private final Resource answerUserPrompt;

    public RAGService(ChatClient chatClient,
                      VectorStore vectorStore,
                      RAGProperties ragProperties,
                      DashscopeRerankService dashscopeRerankService,
                      @Value("classpath:/prompts/rewrite-system.st") Resource rewriteSystemPrompt,
                      @Value("classpath:/prompts/rewrite-user.st") Resource rewriteUserPrompt,
                      @Value("classpath:/prompts/answer-system.st") Resource answerSystemPrompt,
                      @Value("classpath:/prompts/answer-user.st") Resource answerUserPrompt) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
        this.ragProperties = ragProperties;
        this.dashscopeRerankService = dashscopeRerankService;
        this.rewriteSystemPrompt = rewriteSystemPrompt;
        this.rewriteUserPrompt = rewriteUserPrompt;
        this.answerSystemPrompt = answerSystemPrompt;
        this.answerUserPrompt = answerUserPrompt;
    }

    public String rewriteQuestion(String originalQuestion) {
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model(ragProperties.getRewriteModel())
                .temperature(0.1)
                .build();

        String rewrite = chatClient.prompt()
                .system(system -> system.text(rewriteSystemPrompt))
                .user(u -> u.text(rewriteUserPrompt)
                        .param("question", originalQuestion))
                .options(options)
                .call()
                .content();

        if (rewrite == null || rewrite.isBlank()) {
            return originalQuestion;
        }
        return rewrite.trim();
    }

    public List<Document> retrieveCandidates(String rewrittenQuestion, String kb, int topK) {
        SearchRequest.Builder builder = SearchRequest.builder()
                .query(rewrittenQuestion)
                .topK(topK)
                .similarityThresholdAll();

        if (kb != null && !kb.isBlank()) {
            builder.filterExpression("kb == '" + escapeForFilter(kb) + "'");
        }

        return vectorStore.similaritySearch(builder.build());
    }

    public List<Document> rerank(String originalQuestion, List<Document> candidates, int topN) {
        if (candidates == null || candidates.isEmpty()) {
            return List.of();
        }

        int safeTopN = Math.max(1, topN);
        List<Document> validCandidates = new ArrayList<>();
        List<String> candidateTexts = new ArrayList<>();
        for (Document candidate : candidates) {
            String text = safeText(candidate);
            if (text.isBlank()) {
                continue;
            }
            validCandidates.add(candidate);
            candidateTexts.add(truncate(text, ragProperties.getRerankMaxDocumentChars()));
        }

        if (validCandidates.isEmpty()) {
            return List.of();
        }

        try {
            List<RerankItem> rerankResults = dashscopeRerankService.rerank(originalQuestion, candidateTexts, safeTopN);
            List<Document> reranked = pickByRerankResults(validCandidates, rerankResults, safeTopN);
            if (!reranked.isEmpty()) {
                return reranked;
            }
        } catch (Exception ex) {
            log.warn("DashScope rerank failed, fallback to vector score. reason={}", ex.getMessage());
        }

        return fallbackByVectorScore(validCandidates, safeTopN);
    }

    public void streamAnswer(String question, List<Document> rerankedDocs, Consumer<String> tokenConsumer) {
        String context = buildContext(rerankedDocs);
        OpenAiChatOptions options = OpenAiChatOptions.builder()
                .model(ragProperties.getAnswerModel())
                .temperature(0.2)
                .build();

        chatClient.prompt()
                .system(system -> system.text(answerSystemPrompt))
                .user(u -> u.text(answerUserPrompt)
                        .param("question", question)
                        .param("context", context))
                .options(options)
                .stream()
                .content()
                .doOnNext(tokenConsumer)
                .blockLast();
    }

    public List<String> references(List<Document> docs) {
        return docs.stream()
                .map(this::referenceFromMetadata)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
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

    private String buildContext(List<Document> docs) {
        if (docs == null || docs.isEmpty()) {
            return "(无可用知识片段)";
        }

        StringBuilder builder = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i);
            builder.append("[片段").append(i + 1).append("]\n");
            builder.append(safeText(doc)).append("\n");
            String ref = referenceFromMetadata(doc);
            if (ref != null) {
                builder.append("来源：").append(ref).append("\n");
            }
            builder.append("\n");
        }
        return builder.toString();
    }

    private String referenceFromMetadata(Document doc) {
        Map<String, Object> metadata = doc.getMetadata();
        if (metadata == null || metadata.isEmpty()) {
            return null;
        }

        Object source = metadata.get("source");
        if (source == null) {
            source = metadata.get("filename");
        }
        if (source == null) {
            source = metadata.get("file_name");
        }

        Object chunkIndex = metadata.get("chunk_index");
        if (source == null) {
            return null;
        }

        if (chunkIndex == null) {
            return source.toString();
        }
        return source + "#chunk-" + chunkIndex;
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

    private String escapeForFilter(String kb) {
        return kb.replace("'", "\\'");
    }
}
