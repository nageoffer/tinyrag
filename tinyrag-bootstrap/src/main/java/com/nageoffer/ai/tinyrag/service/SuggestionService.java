package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.service.rag.ChatResponseUtils;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.document.Document;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

/**
 * 推荐问题生成服务
 * <p>
 * 基于用户问题做独立向量检索 + 收集 MCP 工具描述，调用 LLM 生成有据可答的推荐问题
 * 该服务无状态、线程安全，可在任意线程池中并行调用
 * 不过有个性能问题，那就是常规的知识问答会调用一次向量库，这里还会重复调用一次向量库。如果是对底层向量库性能敏感，考虑优化
 */
@Slf4j
@Service
public class SuggestionService {

    private final ChatClient suggestionsClient;
    private final VectorStore vectorStore;
    private final ToolCallback[] toolCallbacks;
    private final RAGProperties ragProperties;
    private final Resource systemPrompt;
    private final Resource userPrompt;
    private final ObjectMapper objectMapper = new ObjectMapper();

    public SuggestionService(ChatModel chatModel,
                             VectorStore vectorStore,
                             ToolCallback[] toolCallbacks,
                             RAGProperties ragProperties,
                             @Value("classpath:/prompts/suggestions-system.st") Resource systemPrompt,
                             @Value("classpath:/prompts/suggestions-user.st") Resource userPrompt) {
        this.suggestionsClient = ChatClient.builder(chatModel).build();
        this.vectorStore = vectorStore;
        this.toolCallbacks = toolCallbacks;
        this.ragProperties = ragProperties;
        this.systemPrompt = systemPrompt;
        this.userPrompt = userPrompt;
    }

    /**
     * 根据用户问题和知识库标识，生成推荐的后续问题列表
     *
     * @param question 用户原始问题
     * @param kb       知识库标识（可为空）
     * @return 推荐问题列表，失败时返回空列表
     */
    public List<String> generate(String question, String kb) {
        try {
            // 1. 独立向量检索，获取相关文档
            SearchRequest.Builder searchBuilder = SearchRequest.builder()
                    .query(question)
                    .topK(ragProperties.getRetrieveTopK());
            if (StringUtils.hasText(kb)) {
                searchBuilder.filterExpression("kb == '" + escapeForFilter(kb) + "'");
            }
            List<Document> documents = vectorStore.similaritySearch(searchBuilder.build());

            String docContext = documents.stream()
                    .map(Document::getText)
                    .collect(Collectors.joining("\n---\n"));

            // 2. 收集 MCP 工具描述
            String toolDesc = Arrays.stream(toolCallbacks)
                    .map(tc -> tc.getToolDefinition().name() + ": " + tc.getToolDefinition().description())
                    .collect(Collectors.joining("\n"));

            if (!StringUtils.hasText(docContext) && !StringUtils.hasText(toolDesc)) {
                return List.of();
            }

            // 3. 调用 LLM 生成推荐问题
            ChatClientResponse response = suggestionsClient.prompt()
                    .system(system -> system.text(systemPrompt))
                    .user(user -> user.text(userPrompt)
                            .param("question", question)
                            .param("retrievedDocuments", formatSection("参考文档", docContext))
                            .param("toolDescriptions", formatSection("可用工具", toolDesc)))
                    .options(ChatOptions.builder().temperature(0.7).maxTokens(256).build())
                    .call()
                    .chatClientResponse();

            String result = ChatResponseUtils.extractText(response);
            if (!StringUtils.hasText(result)) {
                return List.of();
            }

            // 4. 解析 JSON 数组
            return parseJsonArray(result.trim());
        } catch (Exception ex) {
            log.warn("[Suggestions] 推荐问题生成失败: {}", ex.getMessage());
            return List.of();
        }
    }

    private List<String> parseJsonArray(String text) {
        try {
            int start = text.indexOf('[');
            int end = text.lastIndexOf(']');
            if (start < 0 || end < 0 || end <= start) {
                return List.of();
            }
            String jsonArray = text.substring(start, end + 1);
            return objectMapper.readValue(jsonArray, new TypeReference<>() {
            });
        } catch (Exception ex) {
            log.warn("[Suggestions] 解析推荐问题JSON失败: {}", ex.getMessage());
            return List.of();
        }
    }

    private String formatSection(String label, String content) {
        return StringUtils.hasText(content) ? "【" + label + "】\n" + content : "";
    }

    private String escapeForFilter(String kb) {
        return kb.replace("'", "\\'");
    }
}
