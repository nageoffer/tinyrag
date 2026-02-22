package com.nageoffer.ai.tinyrag.prompt;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Chunk {

    private String id;           // chunk 唯一 ID
    private String source;       // 来源
    private String updateTime;   // 更新时间
    private String content;      // 内容
}
