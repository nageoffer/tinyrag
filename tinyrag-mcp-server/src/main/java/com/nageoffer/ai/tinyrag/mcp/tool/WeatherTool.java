package com.nageoffer.ai.tinyrag.mcp.tool;

import lombok.extern.slf4j.Slf4j;
import org.springaicommunity.mcp.annotation.McpTool;
import org.springaicommunity.mcp.annotation.McpToolParam;
import org.springframework.stereotype.Component;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * 天气查询工具
 */
@Slf4j
@Component
public class WeatherTool {

    private static final Random RANDOM = new Random();
    private static final String[] WEATHER_CONDITIONS = {"晴", "多云", "阴", "小雨", "中雨", "大雨"};
    private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    @McpTool(
            name = "queryWeatherForecast",
            description = "查询指定城市的天气预报，支持查询1-7天的天气情况"
    )
    public Response queryWeatherForecast(
            @McpToolParam(description = "城市名称，例如：北京、上海、深圳")
            String location,
            @McpToolParam(required = false, description = "查询天数，1-7天，默认为1天")
            Integer days
    ) {
        log.info("查询 {} 未来 {} 天的天气预报", location, days);
        int queryDays = days != null ? days : 1;

        if (queryDays < 1 || queryDays > 7) {
            return new Response("error", "查询天数必须在1-7天之间", null);
        }

        List<DailyWeather> forecast = new ArrayList<>();
        LocalDate today = LocalDate.now();

        for (int i = 0; i < queryDays; i++) {
            LocalDate date = today.plusDays(i);
            String dateStr = date.format(DATE_FORMATTER);
            String condition = WEATHER_CONDITIONS[RANDOM.nextInt(WEATHER_CONDITIONS.length)];
            int tempHigh = 15 + RANDOM.nextInt(20);
            int tempLow = tempHigh - 5 - RANDOM.nextInt(8);

            forecast.add(new DailyWeather(dateStr, condition, tempLow, tempHigh));
        }

        String summary = String.format("%s未来%d天天气预报已生成", location, queryDays);
        return new Response("success", summary, forecast);
    }

    public record DailyWeather(
            String date,
            String condition,
            int tempLow,
            int tempHigh
    ) {
    }

    public record Response(
            String status,
            String message,
            List<DailyWeather> forecast
    ) {
    }
}
