package com.nageoffer.ai.tinyrag.controller;

import com.nageoffer.ai.tinyrag.config.RAGProperties;
import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.model.UploadResponse;
import com.nageoffer.ai.tinyrag.service.KnowledgeIngestionService;
import com.nageoffer.ai.tinyrag.service.RAGService;
import jakarta.validation.Valid;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.springframework.ai.document.Document;
import org.springframework.core.task.TaskExecutor;
import org.springframework.http.MediaType;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

@RestController
@RequestMapping("/api/rag")
@Validated
public class RAGController {

    private final RAGService ragService;
    private final KnowledgeIngestionService ingestionService;
    private final RAGProperties ragProperties;
    private final TaskExecutor taskExecutor;

    public RAGController(RAGService ragService,
                         KnowledgeIngestionService ingestionService,
                         RAGProperties ragProperties,
                         TaskExecutor taskExecutor) {
        this.ragService = ragService;
        this.ingestionService = ingestionService;
        this.ragProperties = ragProperties;
        this.taskExecutor = taskExecutor;
    }

    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamChat(@Valid @RequestBody RAGRequest request) {
        SseEmitter emitter = new SseEmitter(0L);

        taskExecutor.execute(() -> {
            try {
                RAGExecutionContext context = prepareContext(request);
                sendEvent(emitter, "meta", Map.of("rewrittenQuestion", context.rewrittenQuestion()));
                sendEvent(emitter, "refs", Map.of("references", context.references()));

                ragService.streamAnswer(request.getQuestion(), context.rerankedDocs(),
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

    @PostMapping(value = "/knowledge/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public UploadResponse uploadFile(@RequestParam("file") MultipartFile file,
                                     @RequestParam(value = "kb", required = false) String kb) {
        return ingestionService.ingest(file, kb);
    }

    private RAGExecutionContext prepareContext(RAGRequest request) {
        String rewritten = ragService.rewriteQuestion(request.getQuestion());
        String kb = request.getKb();

        List<Document> candidates = ragService.retrieveCandidates(rewritten, kb, ragProperties.getRetrieveTopK());
        List<Document> reranked = ragService.rerank(request.getQuestion(), candidates, ragProperties.getRerankTopN());
        List<String> refs = ragService.references(reranked);
        return new RAGExecutionContext(rewritten, reranked, refs);
    }

    private void sendEvent(SseEmitter emitter, String event, Object data) {
        try {
            emitter.send(SseEmitter.event().name(event).data(data));
        } catch (IOException ex) {
            throw new IllegalStateException("SSE 发送失败", ex);
        }
    }

    private record RAGExecutionContext(String rewrittenQuestion, List<Document> rerankedDocs, List<String> references) {
    }
}
