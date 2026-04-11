package com.nageoffer.ai.tinyrag.config;

import com.nageoffer.ai.tinyrag.service.rag.ElasticsearchDocumentRepository;
import com.nageoffer.ai.tinyrag.service.rag.HybridDocumentRetriever;
import com.nageoffer.ai.tinyrag.service.rag.KeywordDocumentRetriever;
import com.nageoffer.ai.tinyrag.service.rag.RewriteQueryTransformer;
import com.nageoffer.ai.tinyrag.service.rag.NonReturnDirectToolCallback;

import io.modelcontextprotocol.client.McpSyncClient;
import org.apache.tika.Tika;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.InMemoryChatMemoryRepository;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.mcp.SyncMcpToolCallbackProvider;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.generation.augmentation.ContextualQueryAugmenter;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
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
    public ChatMemory chatMemory(RAGProperties ragProperties) {
        return MessageWindowChatMemory.builder()
                .chatMemoryRepository(new InMemoryChatMemoryRepository())
                .maxMessages(ragProperties.getMemoryMaxMessages())
                .build();
    }

    @Bean
    public KeywordDocumentRetriever keywordDocumentRetriever(
            ElasticsearchDocumentRepository esRepository,
            RAGProperties ragProperties) {
        return new KeywordDocumentRetriever(esRepository, ragProperties);
    }

    @Bean
    public HybridDocumentRetriever hybridDocumentRetriever(
            VectorStore vectorStore,
            KeywordDocumentRetriever keywordDocumentRetriever,
            RAGProperties ragProperties) {
        VectorStoreDocumentRetriever vectorRetriever = VectorStoreDocumentRetriever.builder()
                .vectorStore(vectorStore)
                .topK(ragProperties.getRetrieveTopK())
                .build();
        return new HybridDocumentRetriever(vectorRetriever, keywordDocumentRetriever, ragProperties);
    }

    @Bean
    public RetrievalAugmentationAdvisor retrievalAugmentationAdvisor(
            HybridDocumentRetriever hybridDocumentRetriever,
            RewriteQueryTransformer rewriteQueryTransformer,
            List<DocumentPostProcessor> documentPostProcessors,
            @Value("classpath:/prompts/answer-user.st") Resource ragAugmentPrompt) {
        ContextualQueryAugmenter queryAugmenter = ContextualQueryAugmenter.builder()
                .promptTemplate(new PromptTemplate(ragAugmentPrompt))
                .allowEmptyContext(true)
                .build();

        return RetrievalAugmentationAdvisor.builder()
                .queryTransformers(rewriteQueryTransformer)
                .documentRetriever(hybridDocumentRetriever)
                .documentPostProcessors(documentPostProcessors)
                .queryAugmenter(queryAugmenter)
                .build();
    }

    @Bean
    public ChatClient chatClient(ChatModel chatModel,
                                 ToolCallback[] toolCallbacks,
                                 ChatMemory chatMemory,
                                 RetrievalAugmentationAdvisor retrievalAugmentationAdvisor,
                                 @Value("classpath:/prompts/answer-system.st") Resource answerSystemPrompt) {
        MessageChatMemoryAdvisor memoryAdvisor = MessageChatMemoryAdvisor.builder(chatMemory).build();
        return ChatClient.builder(chatModel)
                .defaultSystem(answerSystemPrompt)
                .defaultToolCallbacks(toolCallbacks)
                .defaultAdvisors(memoryAdvisor, retrievalAugmentationAdvisor)
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
    public RewriteQueryTransformer rewriteQueryTransformer(
            ChatModel chatModel,
            RAGProperties ragProperties,
            @Value("classpath:/prompts/rewrite-system.st") Resource rewriteSystemPrompt,
            @Value("classpath:/prompts/rewrite-user.st") Resource rewriteUserPrompt) {
        return new RewriteQueryTransformer(
                chatModel,
                rewriteSystemPrompt,
                rewriteUserPrompt,
                ragProperties.getRewriteModel());
    }

    @Bean
    public TaskExecutor ragTaskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(4);
        executor.setMaxPoolSize(8);
        executor.setQueueCapacity(200);
        executor.setThreadNamePrefix("rag-sse-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.setAwaitTerminationSeconds(30);
        executor.initialize();
        return executor;
    }
}
