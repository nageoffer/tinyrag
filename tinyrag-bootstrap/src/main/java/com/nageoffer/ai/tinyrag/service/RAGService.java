package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.service.rag.ChatResponseUtils;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
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

    public RAGService(ChatClient chatClient,
                      ChatModel chatModel,
                      RAGProperties ragProperties,
                      @Value("classpath:/prompts/title-system.st") Resource titleSystemPrompt,
                      @Value("classpath:/prompts/title-user.st") Resource titleUserPrompt,
                      @Qualifier("ragTaskExecutor") TaskExecutor taskExecutor) {
        this.chatClient = chatClient;
        this.titleClient = ChatClient.builder(chatModel).build();
        this.ragProperties = ragProperties;
        this.titleSystemPrompt = titleSystemPrompt;
        this.titleUserPrompt = titleUserPrompt;
        this.taskExecutor = taskExecutor;
    }

    public SseEmitter streamChat(RAGRequest request) {
        SseEmitter emitter = new SseEmitter(180000L);

        boolean newSession = !StringUtils.hasText(request.getSessionId());
        String sessionId = newSession
                ? UUID.randomUUID().toString()
                : request.getSessionId();

        taskExecutor.execute(() -> {
            try {
                // 新会话第一次提问：生成会话标题
                String sessionTitle = null;
                if (newSession) {
                    sessionTitle = generateTitle(request.getQuestion());
                }

                Map<String, String> meta = new HashMap<>();
                meta.put("sessionId", sessionId);
                if (sessionTitle != null) {
                    meta.put("sessionTitle", sessionTitle);
                }
                sendEvent(emitter, "meta", meta);

                streamAnswer(request.getQuestion(), request.getKb(), sessionId,
                        token -> sendEvent(emitter, "token", token));

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
