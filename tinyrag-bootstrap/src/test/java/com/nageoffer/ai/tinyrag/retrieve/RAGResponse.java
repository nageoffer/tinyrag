package com.nageoffer.ai.tinyrag.retrieve;

import lombok.Data;

import java.util.List;

/**
 * RAG 生成结果，包含回答文本和引用信息
 */
@Data
public class RAGResponse {

    private String answer;                    // 模型的回答（原始文本，带 [1][2] 标记）
    private String renderedAnswer;            // 渲染后的回答（引用标记替换为链接）
    private List<CitationInfo> citations;     // 引用详情列表

    /**
     * 引用信息
     */
    @Data
    public static class CitationInfo {

        private Integer index;           // 引用编号
        private String source;       // 来源文档
        private String sourceUrl;    // 原文链接
        private String chunkContent; // 被引用的 chunk 内容
    }
}
