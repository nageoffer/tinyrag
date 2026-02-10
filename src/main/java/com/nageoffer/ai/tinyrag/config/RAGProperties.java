package com.nageoffer.ai.tinyrag.config;

import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import lombok.Getter;
import lombok.Setter;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Setter
@Getter
@Validated
@ConfigurationProperties(prefix = "app.rag")
public class RAGProperties {

    @NotBlank
    private String rewriteModel;

    @NotBlank
    private String answerModel;

    @NotBlank
    private String rerankModel;

    @NotBlank
    private String rerankEndpoint;

    @Min(1)
    @Max(100)
    private Integer retrieveTopK;

    @Min(1)
    @Max(100)
    private Integer rerankTopN;

    @Min(128)
    @Max(1024000)
    private Integer rerankMaxDocumentChars;

    @Min(128)
    @Max(4000)
    private Integer chunkSize;

    @Min(32)
    @Max(1000)
    private Integer minChunkSizeChars;

    @Min(1)
    @Max(100)
    private Integer minChunkLengthToEmbed;

    @Min(1)
    @Max(10000)
    private Integer maxNumChunks;
}
