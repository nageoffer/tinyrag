package com.nageoffer.ai.tinyrag.config;

import com.nageoffer.ai.tinyrag.service.rag.QueryRewriteService;
import com.nageoffer.ai.tinyrag.service.rag.NonReturnDirectToolCallback;

import io.modelcontextprotocol.client.McpSyncClient;
import org.apache.tika.Tika;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.rag.retrieval.search.DocumentRetriever;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.task.TaskExecutor;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.Arrays;
import java.util.List;

@Configuration
@EnableConfigurationProperties(RAGProperties.class)
public class RAGConfiguration {

    @Bean
    public ToolCallback[] ragMcpToolCallbacks(List<McpSyncClient> mcpSyncClients) {
        SyncMcpToolCallbackProvider toolCallbackProvider = SyncMcpToolCallbackProvider.builder()
                .mcpClients(mcpSyncClients)
                .build();

        return Arrays.stream(toolCallbackProvider.getToolCallbacks())
                .map(NonReturnDirectToolCallback::wrap)
                .toArray(ToolCallback[]::new);
    }

    @Bean
    public ChatClient toolAwareChatClient(ChatModel chatModel,
                                          ToolCallback[] toolCallbacks) {
        return ChatClient.builder(chatModel)
                .defaultToolCallbacks(toolCallbacks)
                .build();
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

    @Bean
    public QueryRewriteService queryRewriteService(
            ChatModel chatModel,
            RAGProperties ragProperties,
            @Value("classpath:/prompts/rewrite-system.st") Resource rewriteSystemPrompt,
            @Value("classpath:/prompts/rewrite-user.st") Resource rewriteUserPrompt) {
        return new QueryRewriteService(
                chatModel,
                rewriteSystemPrompt,
                rewriteUserPrompt,
                ragProperties.getRewriteModel());
    }

    @Bean
    public DocumentRetriever documentRetriever(VectorStore vectorStore, RAGProperties ragProperties) {
        return VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .topK(ragProperties.getRetrieveTopK())
                .similarityThreshold(0.0)
                .build();
    }

    @Bean
    public TaskExecutor ragTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(8);
        executor.setQueueCapacity(200);
        executor.setThreadNamePrefix("rag-sse-");
        executor.initialize();
        return executor;
    }
}
