# 新电脑初始化脚本：换任何电脑，跑一次就绪
# 用法（PowerShell）：
#   git clone https://github.com/Bluemingovo/Blueming-llm-rag-agent
#   cd Blueming-llm-rag-agent
#   powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) {
    Write-Host "[错误] 未找到 Python，请先安装 https://www.python.org/downloads/（勾选 Add to PATH）" -ForegroundColor Red
    exit 1
}
Write-Host "==> Python: $py"

Write-Host "==> 安装依赖..."
python -m pip install -r requirements.txt

Write-Host "==> 配置 .env（需要填 DeepSeek API Key）..."
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "   已创建 .env，请用记事本打开填入 DEEPSEEK_API_KEY=sk-xxx" -ForegroundColor Yellow
} else {
    Write-Host "   .env 已存在，跳过"
}

Write-Host "==> 离线自检..."
$env:PYTHONIOENCODING = "utf-8"
python -m src.selfcheck

Write-Host "==> 拉取共享记忆..."
python -m scripts.fetch_memory

Write-Host ""
Write-Host "✅ 初始化完成！现在可以：" -ForegroundColor Green
Write-Host "   - 启动服务: python -m uvicorn src.app:app --host 127.0.0.1 --port 8000"
Write-Host "   - MCP 接入: 参考 docs/mcp.md（把命令里的路径改成本机路径）"
Write-Host "   - 读记忆:   python -m scripts.fetch_memory"
