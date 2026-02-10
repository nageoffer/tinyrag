# TinyRAG (Spring AI 1.1.2)

一个面向企业内部知识库的最简 RAG Demo，包含：

- 问题重写（Query Rewrite）
- Milvus 向量检索
- 基于百炼原生接口的 Rerank 重排
- 基于百炼兼容接口调用 `Qwen3-Max` 进行流式回复
- 文件上传 + 文档切分 + 向量化入库

## 1. 环境要求

- JDK 17+
- Milvus（默认 `localhost:19530`）
- DashScope API Key（百炼）

## 2. 配置

主要配置在 `src/main/resources/application.yaml`：

- `spring.ai.openai.base-url`: 百炼 OpenAI 兼容地址，默认 `https://dashscope.aliyuncs.com/compatible-mode`
- `spring.ai.openai.api-key`: 百炼 API Key（推荐用环境变量 `DASHSCOPE_API_KEY`）
- `spring.ai.openai.chat.options.model`: 默认聊天模型（`qwen3-max`）
- `spring.ai.openai.embedding.options.model`: 向量模型（示例：`text-embedding-v4`）
- `spring.ai.vectorstore.milvus.*`: Milvus 连接与集合配置
- `app.rag.*`: RAG 参数（重写模型、Rerank 模型、TopK、Chunk 大小等）
- `app.rag.rerank-endpoint`: Rerank 接口地址（默认 `https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank`）
- 提示词模板文件：`src/main/resources/prompts/`（`rewrite-*`、`answer-*`）

### 2.1 Rerank 参数说明（qwen3-rerank）

`qwen3-rerank` 不能走 OpenAI 兼容的 `chat/completions`，需要调用百炼 Rerank 原生接口。  
当前后端发送的核心参数如下：

```json
{
  "model": "qwen3-rerank",
  "query": "用户问题",
  "documents": ["候选文本1", "候选文本2"],
  "top_n": 4,
  "instruct": ""
}
```

## 3. 启动

```bash
./mvnw spring-boot:run
```

启动后打开：`http://localhost:8080/`  
项目内置了一个简单前端页面（无需单独前端项目），可直接进行文件上传和流式问答。

## 4. 接口

### 4.1 上传知识文件并向量化

- URL: `POST /api/rag/knowledge/upload`
- Content-Type: `multipart/form-data`
- 参数：
  - `file`：文件（仅支持 `DOC` / `DOCX` / `PDF` / `MD`）
  - `kb`：知识库名称（可选，默认 `default`）
- 解析方式：`Apache Tika` 文本提取后再切分向量化

示例：

```bash
curl -X POST 'http://localhost:8080/api/rag/knowledge/upload' \
  -F 'file=@./docs/employee-handbook.md' \
  -F 'kb=hr'
```

返回示例：

```json
{
  "fileName": "employee-handbook.md",
  "kb": "hr",
  "chunkCount": 12
}
```

### 4.2 流式 RAG 问答

- URL: `POST /api/rag/chat/stream`
- Content-Type: `application/json`
- 返回：`text/event-stream`
- 服务端实现：`SseEmitter`（非 Flux 返回）

请求示例：

```json
{
  "question": "年假最多可以累计多少天？",
  "kb": "hr"
}
```

curl 示例：

```bash
curl -N -X POST 'http://localhost:8080/api/rag/chat/stream' \
  -H 'Content-Type: application/json' \
  -d '{"question":"年假最多可以累计多少天？","kb":"hr"}'
```

SSE 事件说明：

- `meta`：返回改写后的问题
- `refs`：返回用于回答的片段来源列表
- `token`：模型流式输出 token
- `done`：流结束

## 5. 说明

- 当前 Rerank 使用百炼原生 HTTP 接口调用 `qwen3-rerank`，不再通过 OpenAI 兼容 Chat 接口。
- 上传接口使用默认固定 chunk 参数（`app.rag.chunk-size` 等配置），满足“固定 Chunk 块大小”场景。
- 上传解析已接入 Apache Tika，覆盖 DOC / DOCX / PDF / MD 常见文档场景。
