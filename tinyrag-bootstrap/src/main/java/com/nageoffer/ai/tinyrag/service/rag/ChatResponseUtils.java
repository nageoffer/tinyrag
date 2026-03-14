package com.nageoffer.ai.tinyrag.service.rag;

import org.springframework.ai.chat.client.ChatClientResponse;

public final class ChatResponseUtils {

    public static String extractText(ChatClientResponse response) {
        if (response == null
                || response.chatResponse() == null
                || response.chatResponse().getResult() == null) {
            return null;
        }
        return response.chatResponse().getResult().getOutput().getText();
    }
}
