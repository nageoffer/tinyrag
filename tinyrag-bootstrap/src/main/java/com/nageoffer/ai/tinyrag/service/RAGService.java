package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.service.rag.ChatResponseUtils;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
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
    private final ChatClient titleClient;
    private final RAGProperties ragProperties;
    private final Resource titleSystemPrompt;
    private final Resource titleUserPrompt;
    private final TaskExecutor taskExecutor;
    private final SuggestionService suggestionService;

    public RAGService(ChatClient chatClient,
                      ChatModel chatModel,
                      RAGProperties ragProperties,
                      @Value("classpath:/prompts/title-system.st") Resource titleSystemPrompt,
                      @Value("classpath:/prompts/title-user.st") Resource titleUserPrompt,
                      @Qualifier("ragTaskExecutor") TaskExecutor taskExecutor,
                      SuggestionService suggestionService) {
        this.chatClient = chatClient;
        this.titleClient = ChatClient.builder(chatModel).build();
        this.ragProperties = ragProperties;
        this.titleSystemPrompt = titleSystemPrompt;
        this.titleUserPrompt = titleUserPrompt;
        this.taskExecutor = taskExecutor;
        this.suggestionService = suggestionService;
    }

    public SseEmitter streamChat(RAGRequest request) {
        SseEmitter emitter = new SseEmitter(180000L);

        boolean newSession = !StringUtils.hasText(request.getSessionId());
        String sessionId = newSession
                ? UUID.randomUUID().toString()
                : request.getSessionId();
        log.info("[RAG] sessionId={}, newSession={}, rawSessionId=[{}]", sessionId, newSession, request.getSessionId());

        taskExecutor.execute(() -> {
            try {
                // 立即发送 meta（sessionId），不等标题生成
                sendEvent(emitter, "meta", Map.of("sessionId", sessionId));

                // 新会话：异步生成标题，生成完单独推送
                if (newSession) {
                    CompletableFuture.supplyAsync(() -> generateTitle(request.getQuestion()), taskExecutor)
                            .thenAccept(title -> {
                                log.info("[RAG] 会话标题: {}", title);
                                sendEvent(emitter, "title", Map.of("sessionTitle", title));
                            });
                }

                // 并行生成推荐问题（生成与回答同时进行，但推送在回答完成后）
                CompletableFuture<List<String>> suggestionsFuture = CompletableFuture
                        .supplyAsync(() -> suggestionService.generate(request.getQuestion(), request.getKb()), taskExecutor);

                streamAnswer(request.getQuestion(), request.getKb(), sessionId,
                        token -> sendEvent(emitter, "token", token));

                pushSuggestions(emitter, suggestionsFuture);
                sendEvent(emitter, "done", "[DONE]");
                emitter.complete();
            } catch (RuntimeException ex) {
                if (ex.getCause() instanceof IOException) {
                    return;
                }
                try {
                    sendEvent(emitter, "error", ex.getMessage() == null ? "stream error" : ex.getMessage());
                } catch (Exception ignored) {
                }
                emitter.completeWithError(ex);
            }
        });

        return emitter;
    }

    public void streamAnswer(String question, String kb, String sessionId,
                             Consumer<String> tokenConsumer) {
        try {
            long startTime = System.currentTimeMillis();
            log.info("[RAG] 原始问题: {}", question);

            // ChatClient 已全局配置 Advisor(QueryTransformer→检索→Rerank→上下文增强) + MCP 工具
            // QueryTransformer 负责将问题改写后用于向量检索，LLM 接收用户原始问题
            ChatClient.ChatClientRequestSpec requestSpec = chatClient.prompt()
                    .user(question);

            requestSpec.advisors(spec -> spec.param(ChatMemory.CONVERSATION_ID, sessionId));
            if (StringUtils.hasText(kb)) {
                requestSpec.advisors(spec -> spec.param(
                        VectorStoreDocumentRetriever.FILTER_EXPRESSION,
                        "kb == '" + escapeForFilter(kb) + "'"));
            }
            if (StringUtils.hasText(ragProperties.getAnswerModel())) {
                requestSpec.options(ChatOptions.builder()
                        .model(ragProperties.getAnswerModel())
                        .build());
            }

            log.info("[RAG] 开始流式调用 LLM...");
            long llmStartTime = System.currentTimeMillis();

            // 流式调用，逐 token 推送回答
            requestSpec.stream().chatClientResponse().toStream().forEach(chunk -> {
                String token = ChatResponseUtils.extractText(chunk);
                if (StringUtils.hasText(token)) {
                    tokenConsumer.accept(token);
                }
            });

            log.info("[RAG] LLM 流式调用完成 ({}ms, 总 {}ms)",
                    System.currentTimeMillis() - llmStartTime,
                    System.currentTimeMillis() - startTime);
        } catch (RuntimeException e) {
            if (e.getCause() instanceof IOException) {
                throw e;
            }
            log.error("[RAG] 问答过程出错", e);
            tokenConsumer.accept("处理问题时出错：" + e.getMessage());
        }
    }

    private void pushSuggestions(SseEmitter emitter, CompletableFuture<List<String>> suggestionsFuture) {
        List<String> suggestions;
        try {
            suggestions = suggestionsFuture.get(15, TimeUnit.SECONDS);
        } catch (Exception ex) {
            log.warn("[RAG] 推荐问题获取失败: {}", ex.getMessage());
            suggestions = List.of();
        }
        if (suggestions.isEmpty()) {
            sendEvent(emitter, "suggestions", Map.of("fallback", "暂无推荐问题"));
        } else {
            log.info("[RAG] 推荐问题: {}", suggestions);
            sendEvent(emitter, "suggestions", Map.of("questions", suggestions));
        }
    }

    private void sendEvent(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data));
        } catch (IOException ex) {
            log.info("[SSE] 客户端已断开连接，停止推送");
            throw new RuntimeException(ex);
        }
    }

    private String escapeForFilter(String kb) {
        return kb.replace("'", "\\'");
    }

    private String generateTitle(String question) {
        try {
            ChatClientResponse response = titleClient.prompt()
                    .system(system -> system.text(titleSystemPrompt))
                    .user(user -> user.text(titleUserPrompt).param("question", question))
                    .options(ChatOptions.builder().temperature(0.0).maxTokens(32).build())
                    .call()
                    .chatClientResponse();

            String title = ChatResponseUtils.extractText(response);
            return StringUtils.hasText(title) ? title.trim() : question;
        } catch (Exception ex) {
            log.warn("生成会话标题失败，回退原问题: {}", ex.getMessage());
            return question;
        }
    }
}
