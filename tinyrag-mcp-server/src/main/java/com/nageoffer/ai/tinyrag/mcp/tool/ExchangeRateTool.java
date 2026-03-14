package com.nageoffer.ai.tinyrag.mcp.tool;

import lombok.extern.slf4j.Slf4j;
import org.springaicommunity.mcp.annotation.McpTool;
import org.springaicommunity.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Component;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.HashMap;
import java.util.Map;

/**
 * 汇率查询工具
 */
@Slf4j
@Component
public class ExchangeRateTool {

    // 模拟汇率数据（以人民币CNY为基准）
    private static final Map<String, BigDecimal> EXCHANGE_RATES = new HashMap<>();

    static {
        EXCHANGE_RATES.put("CNY", BigDecimal.ONE);           // 人民币
        EXCHANGE_RATES.put("USD", new BigDecimal("0.14"));   // 美元
        EXCHANGE_RATES.put("EUR", new BigDecimal("0.13"));   // 欧元
        EXCHANGE_RATES.put("GBP", new BigDecimal("0.11"));   // 英镑
        EXCHANGE_RATES.put("JPY", new BigDecimal("21.50"));  // 日元
        EXCHANGE_RATES.put("HKD", new BigDecimal("1.09"));   // 港币
        EXCHANGE_RATES.put("KRW", new BigDecimal("192.50")); // 韩元
    }

    @McpTool(
            name = "queryExchangeRate",
            description = "查询货币汇率，支持人民币、美元、欧元、英镑、日元、港币、韩元等货币之间的汇率转换"
    )
    public Response queryExchangeRate(
            @McpToolParam(description = "源货币代码，例如：CNY、USD、EUR、GBP、JPY、HKD、KRW")
            String from,
            @McpToolParam(description = "目标货币代码，例如：CNY、USD、EUR、GBP、JPY、HKD、KRW")
            String to,
            @McpToolParam(required = false, description = "金额，默认为1")
            BigDecimal amount
    ) {
        log.info("查询汇率：{} -> {}", from, to);
        String fromUpper = from.toUpperCase();
        String toUpper = to.toUpperCase();
        BigDecimal queryAmount = amount != null ? amount : BigDecimal.ONE;

        if (!EXCHANGE_RATES.containsKey(fromUpper)) {
            return new Response("error", "不支持的源货币: " + from, null, null, null, null);
        }

        if (!EXCHANGE_RATES.containsKey(toUpper)) {
            return new Response("error", "不支持的目标货币: " + to, null, null, null, null);
        }

        // 计算汇率：先转换为CNY，再转换为目标货币
        BigDecimal fromRate = EXCHANGE_RATES.get(fromUpper);
        BigDecimal toRate = EXCHANGE_RATES.get(toUpper);
        BigDecimal rate = toRate.divide(fromRate, 6, RoundingMode.HALF_UP);
        BigDecimal result = queryAmount.multiply(rate).setScale(2, RoundingMode.HALF_UP);

        String message = String.format("%s %s = %s %s (汇率: 1 %s = %s %s)",
                queryAmount, fromUpper, result, toUpper, fromUpper, rate.setScale(4, RoundingMode.HALF_UP), toUpper);

        return new Response("success", message, fromUpper, toUpper, rate, result);
    }

    public record Response(
            String status,
            String message,
            String from,
            String to,
            BigDecimal rate,
            BigDecimal result
    ) {
    }
}
