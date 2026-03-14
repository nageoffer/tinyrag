package com.nageoffer.ai.tinyrag.service.rag;

import org.springframework.ai.chat.model.ToolContext;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.ToolDefinition;
import org.springframework.ai.tool.metadata.ToolMetadata;
import org.springframework.util.Assert;

public final class NonReturnDirectToolCallback implements ToolCallback {

    private static final ToolMetadata NON_RETURN_DIRECT_METADATA =
            ToolMetadata.builder().returnDirect(false).build();

    private final ToolCallback delegate;

    private NonReturnDirectToolCallback(ToolCallback delegate) {
        this.delegate = delegate;
    }

    public static ToolCallback wrap(ToolCallback toolCallback) {
        Assert.notNull(toolCallback, "toolCallback must not be null");
        if (!toolCallback.getToolMetadata().returnDirect()) {
            return toolCallback;
        }
        return new NonReturnDirectToolCallback(toolCallback);
    }

    @Override
    public ToolDefinition getToolDefinition() {
        return delegate.getToolDefinition();
    }

    @Override
    public ToolMetadata getToolMetadata() {
        return NON_RETURN_DIRECT_METADATA;
    }

    @Override
    public String call(String toolInput) {
        return delegate.call(toolInput);
    }

    @Override
    public String call(String toolInput, ToolContext toolContext) {
        return delegate.call(toolInput, toolContext);
    }
}
