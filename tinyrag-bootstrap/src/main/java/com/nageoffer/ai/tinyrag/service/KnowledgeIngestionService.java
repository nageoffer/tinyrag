package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.model.UploadResponse;
import com.nageoffer.ai.tinyrag.service.rag.ElasticsearchDocumentRepository;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.apache.tika.Tika;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import static org.springframework.http.HttpStatus.BAD_REQUEST;

@Slf4j
@Service
@RequiredArgsConstructor
public class KnowledgeIngestionService {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("pdf", "doc", "docx", "md", "markdown");

    private final VectorStore vectorStore;
    private final TokenTextSplitter tokenTextSplitter;
    private final Tika tika;
    private final ElasticsearchDocumentRepository esRepository;

    public UploadResponse ingest(MultipartFile file, String kb) {
        String fileName = normalizeFileName(file.getOriginalFilename());
        validateFileType(fileName);
        String kbName = normalizeKb(kb);

        log.info("[Ingest] 开始处理文件: {}, kb={}", fileName, kbName);
        String content = parseContentWithTika(file);

        if (!StringUtils.hasText(content)) {
            throw new ResponseStatusException(BAD_REQUEST, "上传文件内容为空");
        }
        log.info("[Ingest] Tika 解析完成, 内容长度: {} 字符", content.length());

        List<Document> chunks = splitToDocuments(content, fileName, kbName);
        if (chunks.isEmpty()) {
            throw new ResponseStatusException(BAD_REQUEST, "文件切分后无有效文本");
        }
        log.info("[Ingest] 文本切分完成, 共 {} 个 chunk", chunks.size());

        vectorStore.add(chunks);
        esRepository.indexDocuments(chunks);
        log.info("[Ingest] 入库完成: file={}, kb={}, chunks={}", fileName, kbName, chunks.size());
        return new UploadResponse(fileName, kbName, chunks.size());
    }

    private List<Document> splitToDocuments(String content, String fileName, String kb) {
        String extension = getFileExtension(fileName);

        Document sourceDoc = Document.builder()
                .text(content)
                .metadata("source", fileName)
                .metadata("filename", fileName)
                .metadata("kb", kb)
                .metadata("file_type", extension)
                .build();

        List<Document> splitDocs = tokenTextSplitter.split(sourceDoc);
        List<Document> chunks = new ArrayList<>(splitDocs.size());

        for (int i = 0; i < splitDocs.size(); i++) {
            Document doc = splitDocs.get(i);
            Map<String, Object> metadata = new HashMap<>(doc.getMetadata());
            metadata.put("chunk_index", i);

            chunks.add(Document.builder()
                    .text(doc.getText())
                    .metadata(metadata)
                    .build());
        }

        return chunks;
    }

    private String normalizeFileName(String fileName) {
        if (!StringUtils.hasText(fileName)) {
            throw new ResponseStatusException(BAD_REQUEST, "文件名不能为空");
        }
        return fileName.trim();
    }

    private String normalizeKb(String kb) {
        if (!StringUtils.hasText(kb)) {
            return "default";
        }
        return kb.trim();
    }

    private void validateFileType(String fileName) {
        String extension = getFileExtension(fileName);
        if (!SUPPORTED_EXTENSIONS.contains(extension)) {
            throw new ResponseStatusException(BAD_REQUEST, "仅支持上传 DOC、DOCX、PDF、MD 文件");
        }
    }

    private String getFileExtension(String fileName) {
        String extension = StringUtils.getFilenameExtension(fileName);
        if (!StringUtils.hasText(extension)) {
            return "";
        }
        return extension.toLowerCase(Locale.ROOT);
    }

    private String parseContentWithTika(MultipartFile file) {
        try (InputStream inputStream = file.getInputStream()) {
            String content = tika.parseToString(inputStream);
            return content == null ? "" : content.trim();
        } catch (Exception ex) {
            throw new ResponseStatusException(BAD_REQUEST, "使用 Tika 解析文件失败", ex);
        }
    }
}
