package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.service.rag.QueryRewriteService;
import com.nageoffer.ai.tinyrag.service.rag.RerankDocumentPostProcessor;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.stream.Stream;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.task.TaskExecutor;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@Slf4j
@Service
public class RAGService {

    private final ChatClient chatClient;
    private final RAGProperties ragProperties;
    private final QueryRewriteService queryRewriteService;
    private final DocumentRetriever documentRetriever;
    private final RerankDocumentPostProcessor rerankPostProcessor;
    private final TaskExecutor taskExecutor;
    private final Resource answerSystemPrompt;
    private final Resource answerUserPrompt;

    public RAGService(ChatClient chatClient,
                      RAGProperties ragProperties,
                      QueryRewriteService queryRewriteService,
                      DocumentRetriever documentRetriever,
                      RerankDocumentPostProcessor rerankPostProcessor,
                      @Qualifier("ragTaskExecutor") TaskExecutor taskExecutor,
                      @Value("classpath:/prompts/answer-system.st") Resource answerSystemPrompt,
                      @Value("classpath:/prompts/answer-user.st") Resource answerUserPrompt) {
        this.chatClient = chatClient;
        this.ragProperties = ragProperties;
        this.queryRewriteService = queryRewriteService;
        this.documentRetriever = documentRetriever;
        this.rerankPostProcessor = rerankPostProcessor;
        this.taskExecutor = taskExecutor;
        this.answerSystemPrompt = answerSystemPrompt;
        this.answerUserPrompt = answerUserPrompt;
    }

    public SseEmitter streamChat(RAGRequest request) {
        SseEmitter emitter = new SseEmitter(0L);

        taskExecutor.execute(() -> {
            try {
                streamAnswer(request.getQuestion(), request.getKb(),
                        rewrittenQuestion -> sendEvent(emitter, "meta", Map.of("rewrittenQuestion", rewrittenQuestion)),
                        refs -> sendEvent(emitter, "refs", Map.of("references", refs)),
                        token -> sendEvent(emitter, "token", token));

                sendEvent(emitter, "done", "[DONE]");
                emitter.complete();
            } catch (Exception ex) {
                try {
                    sendEvent(emitter, "error", ex.getMessage() == null ? "stream error" : ex.getMessage());
                } catch (Exception ignored) {
                }
                emitter.completeWithError(ex);
            }
        });

        return emitter;
    }

    public void streamAnswer(String question, String kb,
                             Consumer<String> rewrittenQuestionConsumer,
                             Consumer<List<String>> referencesConsumer,
                             Consumer<String> tokenConsumer) {
        try {
            long startTime = System.currentTimeMillis();

            // 1. 问题重写
            String rewritten = queryRewriteService.rewrite(question);
            log.info("[RAG] 原始问题: {}", question);
            log.info("[RAG] 重写问题: {} ({}ms)", rewritten, System.currentTimeMillis() - startTime);
            rewrittenQuestionConsumer.accept(rewritten);

            // 2. 向量检索 → Rerank（直接用重写后的问题，不再做查询扩展）
            List<Document> documents = retrieveDocuments(rewritten, kb);
            log.info("[RAG] 检索+Rerank 完成, {} 个文档 ({}ms)", documents.size(), System.currentTimeMillis() - startTime);
            referencesConsumer.accept(extractReferences(documents));

            // 3. 流式生成：LLM 自动决策是否调用 MCP 工具
            String context = buildContext(documents);
            log.debug("[RAG] 知识库上下文长度: {} 字符", context.length());

            ChatClient.ChatClientRequestSpec requestSpec = buildRequest(question, context);

            log.info("[RAG] 开始流式调用 LLM...");
            long llmStartTime = System.currentTimeMillis();
            int[] tokenCount = {0};
            try (Stream<String> tokenStream = requestSpec.stream().content().toStream()) {
                tokenStream.forEach(token -> {
                    if (StringUtils.hasText(token)) {
                        tokenCount[0]++;
                        tokenConsumer.accept(token);
                    }
                });
            }
            log.info("[RAG] 流式完成, {} 个 token (LLM {}ms, 总 {}ms)",
                    tokenCount[0],
                    System.currentTimeMillis() - llmStartTime,
                    System.currentTimeMillis() - startTime);

            // 4. Fallback：流式返回 0 token（工具调用场景下 Spring AI stream 的已知问题）
            if (tokenCount[0] == 0) {
                log.warn("[RAG] 流式输出 0 token, 降级为同步调用...");
                long fallbackStart = System.currentTimeMillis();
                ChatClient.ChatClientRequestSpec fallbackSpec = buildRequest(question, context);
                ChatClientResponse response = fallbackSpec.call().chatClientResponse();
                String answer = extractContent(response);
                if (StringUtils.hasText(answer)) {
                    log.info("[RAG] 同步调用成功, {} 字符 ({}ms)", answer.length(), System.currentTimeMillis() - fallbackStart);
                    tokenConsumer.accept(answer);
                } else {
                    log.warn("[RAG] 同步调用也未返回内容 ({}ms)", System.currentTimeMillis() - fallbackStart);
                    tokenConsumer.accept("在当前会话中模型未返回文本内容，请重试。");
                }
            }
        } catch (Exception e) {
            log.error("[RAG] 问答过程出错", e);
            tokenConsumer.accept("处理问题时出错：" + e.getMessage());
        }
    }

    private ChatClient.ChatClientRequestSpec buildRequest(String question, String context) {
        ChatClient.ChatClientRequestSpec requestSpec = chatClient.prompt()
                .system(system -> system.text(answerSystemPrompt))
                .user(user -> user.text(answerUserPrompt)
                        .param("question", question)
                        .param("context", context));
        if (StringUtils.hasText(ragProperties.getAnswerModel())) {
            requestSpec.options(ChatOptions.builder()
                    .model(ragProperties.getAnswerModel())
                    .build());
        }
        return requestSpec;
    }

    private List<Document> retrieveDocuments(String rewrittenQuestion, String kb) {
        Query query = new Query(rewrittenQuestion);
        Query queryWithFilter = applyFilter(query, kb);

        List<Document> retrieved = documentRetriever.retrieve(queryWithFilter);
        log.info("[RAG] 向量检索到 {} 个文档", retrieved.size());

        if (retrieved.isEmpty()) {
            return List.of();
        }

        List<Document> reranked = rerankPostProcessor.process(query, new ArrayList<>(retrieved));
        log.info("[RAG] Rerank 后保留 {} 个文档", reranked.size());
        return reranked;
    }

    private Query applyFilter(Query query, String kb) {
        if (!StringUtils.hasText(kb)) {
            return query;
        }
        String filterExpression = "kb == '" + escapeForFilter(kb) + "'";
        return query.mutate()
                .context(Map.of(VectorStoreDocumentRetriever.FILTER_EXPRESSION, filterExpression))
                .build();
    }

    private String buildContext(List<Document> documents) {
        if (documents.isEmpty()) {
            return "无相关知识库片段。";
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < documents.size(); i++) {
            sb.append("[").append(i + 1).append("] ").append(documents.get(i).getText()).append("\n\n");
        }
        return sb.toString().trim();
    }

    private String extractContent(ChatClientResponse response) {
        if (response == null
                || response.chatResponse() == null
                || response.chatResponse().getResult() == null) {
            return null;
        } else {
            response.chatResponse().getResult();
        }
        return response.chatResponse().getResult().getOutput().getText();
    }

    private void sendEvent(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data));
        } catch (IOException ex) {
            throw new IllegalStateException("SSE 发送失败", ex);
        }
    }

    public List<String> extractReferences(List<Document> docs) {
        return docs.stream()
                .map(this::referenceFromMetadata)
                .filter(Objects::nonNull)
                .distinct()
                .toList();
    }

    private String referenceFromMetadata(Document doc) {
        Map<String, Object> metadata = doc.getMetadata();
        if (metadata.isEmpty()) {
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

    private String escapeForFilter(String kb) {
        return kb.replace("'", "\\'");
    }
}
