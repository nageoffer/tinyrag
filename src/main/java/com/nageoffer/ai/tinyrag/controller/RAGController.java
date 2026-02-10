package com.nageoffer.ai.tinyrag.controller;

import com.nageoffer.ai.tinyrag.model.RAGRequest;
import com.nageoffer.ai.tinyrag.model.UploadResponse;
import com.nageoffer.ai.tinyrag.service.KnowledgeIngestionService;
import com.nageoffer.ai.tinyrag.service.RAGService;
import jakarta.validation.Valid;
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

    public RAGController(RAGService ragService,
                         KnowledgeIngestionService ingestionService) {
        this.ragService = ragService;
        this.ingestionService = ingestionService;
    }

    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamChat(@Valid @RequestBody RAGRequest request) {
        return ragService.streamChat(request);
    }

    @PostMapping(value = "/knowledge/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public UploadResponse uploadFile(@RequestParam("file") MultipartFile file,
                                     @RequestParam(value = "kb", required = false) String kb) {
        return ingestionService.ingest(file, kb);
    }
}
