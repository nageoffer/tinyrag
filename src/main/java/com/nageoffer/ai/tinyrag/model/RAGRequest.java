package com.nageoffer.ai.tinyrag.model;

import jakarta.validation.constraints.NotBlank;

public class RAGRequest {

    @NotBlank
    private String question;

    private String kb;

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public String getKb() {
        return kb;
    }

    public void setKb(String kb) {
        this.kb = kb;
    }
}
