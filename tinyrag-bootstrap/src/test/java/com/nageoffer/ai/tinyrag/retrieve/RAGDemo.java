package com.nageoffer.ai.tinyrag.retrieve;

import java.util.List;

public class RAGDemo {

    public static void main(String[] args) throws Exception {
        RAGGenerationService service = new RAGGenerationService();

        // 模拟检索 + 重排序后的 Top-3 chunk
        List<RetrievedChunk> chunks = List.of(
                new RetrievedChunk(
                        "iPhone 16 Pro Max 因屏幕定制工艺，拆封后不支持七天无理由退货。如需退货，需经售后检测确认存在质量问题。",
                        "退货政策文档", "/docs/return-policy", "2026-01-15", 0.95
                ),
                new RetrievedChunk(
                        "标准商品在签收后 7 天内可申请无理由退货，商品需保持完好，不影响二次销售。",
                        "通用退货规则", "/docs/general-return", "2026-01-10", 0.82
                ),
                new RetrievedChunk(
                        "质量问题退货，运费由商家承担；非质量问题退货，运费由买家承担。",
                        "退货运费规则", "/docs/return-shipping", "2026-02-01", 0.78
                )
        );

        String userQuery = "iPhone 16 Pro Max 拆封后还能退吗？运费谁出？";

        // 执行 RAG 生成
        RAGResponse response = service.generate(chunks, userQuery);

        // 输出结果
        System.out.println("=== 用户问题 ===");
        System.out.println(userQuery);
        System.out.println();
        System.out.println("=== 模型回答 ===");
        System.out.println(response.getAnswer());
        System.out.println();
        System.out.println("=== 引用来源 ===");
        for (RAGResponse.CitationInfo citation : response.getCitations()) {
            System.out.printf("[%d] %s（%s）%n",
                    citation.getIndex(), citation.getSource(), citation.getSourceUrl());
        }
    }
}
