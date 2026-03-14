package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.service.rag.QueryRewriteService;

import java.io.IOException;
import java.util.Map;
import java.util.function.Consumer;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.beans.factory.annotation.Qualifier;
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
    private final TaskExecutor taskExecutor;

    public RAGService(ChatClient chatClient,
                      RAGProperties ragProperties,
                      QueryRewriteService queryRewriteService,
                      @Qualifier("ragTaskExecutor") TaskExecutor taskExecutor) {
        this.chatClient = chatClient;
        this.ragProperties = ragProperties;
        this.queryRewriteService = queryRewriteService;
        this.taskExecutor = taskExecutor;
    }

    public SseEmitter streamChat(RAGRequest request) {
        SseEmitter emitter = new SseEmitter(0L);

        taskExecutor.execute(() -> {
            try {
                streamAnswer(request.getQuestion(), request.getKb(),
                        rewrittenQuestion -> sendEvent(emitter, "meta", Map.of("rewrittenQuestion", rewrittenQuestion)),
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
                             Consumer<String> tokenConsumer) {
        try {
            long startTime = System.currentTimeMillis();

            // 1. 问题重写
            String rewritten = queryRewriteService.rewrite(question);
            log.info("[RAG] 原始问题: {}", question);
            log.info("[RAG] 重写问题: {} ({}ms)", rewritten, System.currentTimeMillis() - startTime);
            rewrittenQuestionConsumer.accept(rewritten);

            // 2. ChatClient 已全局配置 Advisor(检索→Rerank→上下文增强) + MCP 工具
            //    这里只需要构建 prompt 并调用，其余全部由框架处理
            ChatClient.ChatClientRequestSpec requestSpec = chatClient.prompt()
                    .user(rewritten);

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

            // 3. 流式调用，逐 token 推送回答
            requestSpec.stream().chatClientResponse().toStream().forEach(chunk -> {
                String token = extractContent(chunk);
                if (StringUtils.hasText(token)) {
                    tokenConsumer.accept(token);
                }
            });

            log.info("[RAG] LLM 流式调用完成 ({}ms, 总 {}ms)",
                    System.currentTimeMillis() - llmStartTime,
                    System.currentTimeMillis() - startTime);
        } catch (Exception e) {
            log.error("[RAG] 问答过程出错", e);
            tokenConsumer.accept("处理问题时出错：" + e.getMessage());
        }
    }

    private String extractContent(ChatClientResponse response) {
        if (response == null
                || response.chatResponse() == null
                || response.chatResponse().getResult() == null) {
            return null;
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

    private String escapeForFilter(String kb) {
        return kb.replace("'", "\\'");
    }
}
