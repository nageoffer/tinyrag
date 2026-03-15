package com.nageoffer.ai.tinyrag.service.rag;

import lombok.extern.slf4j.Slf4j;
import org.jspecify.annotations.NonNull;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.preretrieval.query.transformation.QueryTransformer;
import org.springframework.core.io.Resource;
import org.springframework.util.StringUtils;

/**
 * 检索查询转换器。
 * <p>
 * 将用户原始问题改写为更适合向量检索的表达，仅作用于检索阶段，
 * 不影响发送给 LLM 的用户消息。
 */
@Slf4j
public class RewriteQueryTransformer implements QueryTransformer {

    private static final int REWRITE_MAX_TOKENS = 64;

    private final ChatClient chatClient;
    private final Resource rewriteSystemPrompt;
    private final Resource rewriteUserPrompt;
    private final String rewriteModel;

    public RewriteQueryTransformer(ChatModel chatModel,
                                   Resource rewriteSystemPrompt,
                                   Resource rewriteUserPrompt,
                                   String rewriteModel) {
        this.chatClient = ChatClient.builder(chatModel).build();
        this.rewriteSystemPrompt = rewriteSystemPrompt;
        this.rewriteUserPrompt = rewriteUserPrompt;
        this.rewriteModel = rewriteModel;
    }

    @Override
    public @NonNull Query transform(Query query) {
        String rewritten = rewrite(query.text());
        return query.mutate().text(rewritten).build();
    }

    private String rewrite(String question) {
        if (!StringUtils.hasText(question)) {
            return question;
        }

        try {
            ChatClientResponse response = chatClient.prompt()
                    .system(system -> system.text(rewriteSystemPrompt))
                    .user(user -> user.text(rewriteUserPrompt)
                            .param("question", question))
                    .options(rewriteOptions())
                    .call()
                    .chatClientResponse();

            String rewrite = ChatResponseUtils.extractText(response);
            String finalRewritten = StringUtils.hasText(rewrite) ? rewrite.trim() : question;
            log.info("重写后问题: {}", finalRewritten);
            return finalRewritten;
        } catch (Exception ex) {
            log.warn("问题重写失败，回退原问题: {}", ex.getMessage());
            return question;
        }
    }

    private ChatOptions rewriteOptions() {
        ChatOptions.Builder builder = ChatOptions.builder()
                .temperature(0.0)
                .maxTokens(REWRITE_MAX_TOKENS);
        if (StringUtils.hasText(rewriteModel)) {
            builder.model(rewriteModel);
        }
        return builder.build();
    }
}
