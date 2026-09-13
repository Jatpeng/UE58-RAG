# 07 云端 Embedding 与 LLM（可选）

本项目默认使用本地 Qwen Embedding，不需要 API Key。云端服务是可选方案，适合没有 GPU 或需要更快批处理的情况。

## 1. OpenAI Embedding

安装 SDK：

`python -m pip install openai`

设置当前 PowerShell 会话的密钥：

`$env:OPENAI_API_KEY = "你的API Key"`

调用示例：

`python -c "from openai import OpenAI; c=OpenAI(); r=c.embeddings.create(model='text-embedding-3-small', input='UE character movement', dimensions=1024); print(len(r.data[0].embedding))"`

如果使用 OpenAI 向量，文档和查询必须使用同一个模型和维度。修改模型后需要重新生成全部向量并重建 Qdrant。

## 2. Qwen 云端 Embedding

阿里云 OpenSearch 提供 Qwen3 0.6B 文本向量服务，服务 ID 为：

`ops-qwen3-embedding-0.6b`

请求需要 Host、Workspace、API Key，并将文本放入 `input` 数组。服务返回 1024 维向量。

PowerShell 请求示例：

`$body = @{ input = @('UE character movement'); input_type = 'query' } | ConvertTo-Json`
`Invoke-RestMethod -Method Post -Uri "$env:OPENSEARCH_HOST/v3/openapi/workspaces/$env:OPENSEARCH_WORKSPACE/text-embedding/ops-qwen3-embedding-0.6b" -Headers @{ Authorization = "Bearer $env:OPENSEARCH_API_KEY" } -ContentType 'application/json' -Body $body`

云端 API 每次发送的文本条数、单条长度和请求体大小都有限制，需要在 Provider 中实现批量、重试、限速和失败记录。

## 3. 生成式 LLM

Embedding 只负责把文本变成向量；如果需要“检索后自动回答”，还需要调用一个生成式 LLM：

`用户问题 → RAG 检索 → 拼接上下文 → LLM 生成答案`

LLM 层应保留来源文件、符号和 URL，让回答可以追溯。不要把没有检索证据的内容当作确定事实。

## 4. 安全注意事项

- 不要把 API Key 发到聊天中。
- 不要把 Key 写入 Git、日志或公开配置。
- 云端 API 会接收发送的源码，商业项目应先确认隐私与合规要求。
- 为 API 设置预算、限额、超时和重试上限。

当前项目的本地 Provider 只支持 `provider: qwen`。若要使用 OpenAI 或 OpenSearch，需要增加对应 Provider，再执行 Embedding 和 Qdrant 导入流程。
