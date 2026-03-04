package com.nageoffer.ai.tinyrag.service;

import com.nageoffer.ai.tinyrag.model.UploadResponse;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.apache.tika.Tika;
import org.springframework.ai.document.Document;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import static org.springframework.http.HttpStatus.BAD_REQUEST;

@Service
public class KnowledgeIngestionService {

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of("pdf", "doc", "docx", "md", "markdown");

    private final VectorStore vectorStore;
    private final TokenTextSplitter tokenTextSplitter;
    private final Tika tika;

    public KnowledgeIngestionService(VectorStore vectorStore, TokenTextSplitter tokenTextSplitter, Tika tika) {
        this.vectorStore = vectorStore;
        this.tokenTextSplitter = tokenTextSplitter;
        this.tika = tika;
    }

    public UploadResponse ingest(MultipartFile file, String kb) {
        String fileName = normalizeFileName(file.getOriginalFilename());
        validateFileType(fileName);
        String kbName = normalizeKb(kb);
        String content = parseContentWithTika(file);

        if (!StringUtils.hasText(content)) {
            throw new ResponseStatusException(BAD_REQUEST, "上传文件内容为空");
        }

        List<Document> chunks = splitToDocuments(content, fileName, kbName);
        if (chunks.isEmpty()) {
            throw new ResponseStatusException(BAD_REQUEST, "文件切分后无有效文本");
        }

        vectorStore.add(chunks);
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
            metadata.put("source", fileName);
            metadata.put("filename", fileName);
            metadata.put("kb", kb);
            metadata.put("file_type", extension);
            metadata.put("chunk_index", i);

            Document withMetadata = Document.builder()
                    .text(doc.getText())
                    .metadata(metadata)
                    .build();
            chunks.add(withMetadata);
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
