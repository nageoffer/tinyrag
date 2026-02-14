package com.nageoffer.ai.tinyrag.model;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class UploadResponse {

    private String fileName;

    private String kb;

    private Integer chunkCount;

    public UploadResponse(String fileName, String kb, Integer chunkCount) {
        this.fileName = fileName;
        this.kb = kb;
        this.chunkCount = chunkCount;
    }
}
