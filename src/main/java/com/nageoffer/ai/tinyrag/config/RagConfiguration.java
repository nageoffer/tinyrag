package com.nageoffer.ai.tinyrag.config;

import org.apache.tika.Tika;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableConfigurationProperties(RAGProperties.class)
public class RAGConfiguration {

    @Bean
    public ChatClient chatClient(ChatModel chatModel) {
        return ChatClient.create(chatModel);
    }

    @Bean
    public TokenTextSplitter tokenTextSplitter(RAGProperties ragProperties) {
        return TokenTextSplitter.builder()
                .withChunkSize(ragProperties.getChunkSize())
                .withMinChunkSizeChars(ragProperties.getMinChunkSizeChars())
                .withMinChunkLengthToEmbed(ragProperties.getMinChunkLengthToEmbed())
                .withMaxNumChunks(ragProperties.getMaxNumChunks())
                .withKeepSeparator(true)
                .build();
    }

    @Bean
    public Tika tika() {
        return new Tika();
    }
}
