package com.nageoffer.ai.tinyrag.model;

import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class UploadResponse {

    private String fileName;

    private String kb;

    private int chunkCount;

    public UploadResponse(String fileName, String kb, int chunkCount) {
        this.fileName = fileName;
        this.kb = kb;
        this.chunkCount = chunkCount;
    }
}
