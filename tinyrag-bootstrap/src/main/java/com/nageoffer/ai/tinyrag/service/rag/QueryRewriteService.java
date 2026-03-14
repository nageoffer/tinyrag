package com.nageoffer.ai.tinyrag.service.rag;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.core.io.Resource;
import org.springframework.util.StringUtils;

@Slf4j
public class QueryRewriteService {

    private static final int REWRITE_MAX_TOKENS = 64;

    private final ChatClient chatClient;
    private final Resource rewriteSystemPrompt;
    private final Resource rewriteUserPrompt;
    private final String rewriteModel;

    public QueryRewriteService(ChatModel chatModel,
                               Resource rewriteSystemPrompt,
                               Resource rewriteUserPrompt,
                               String rewriteModel) {
        this.chatClient = ChatClient.builder(chatModel).build();
        this.rewriteSystemPrompt = rewriteSystemPrompt;
        this.rewriteUserPrompt = rewriteUserPrompt;
        this.rewriteModel = rewriteModel;
    }

    public String rewrite(String question) {
        if (!StringUtils.hasText(question)) {
            return question;
        }

        try {
            ChatClient.ChatClientRequestSpec requestSpec = chatClient.prompt()
                    .system(system -> system.text(rewriteSystemPrompt))
                    .user(user -> user.text(rewriteUserPrompt)
                            .param("question", question))
                    .options(rewriteOptions());

            ChatClientResponse rewriteResponse = requestSpec.call().chatClientResponse();

            String rewrite = extractContent(rewriteResponse);
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

    private String extractContent(ChatClientResponse response) {
        if (response == null
                || response.chatResponse() == null
                || response.chatResponse().getResult() == null) {
            return null;
        }
        return response.chatResponse().getResult().getOutput().getText();
    }
}
