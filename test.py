from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch

DEVICE = 'cuda'
DEVICE_ID ='0'
CUDA_DEVICE = f'{DEVICE}:{DEVICE_ID}' if DEVICE_ID else DEVICE

def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

app = FastAPI()

@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history') 
    max_lenth = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')

    response, history = model.chat(
        tokenizer, prompt, history = history, max_lenth = max_lenth if max_lenth else 2048,
        top_p = top_p if top_p else 0.7, temperature = temperature if temperature else 0.95
    )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%D %H:%M:%S")

    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }

    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)  # 打印日志
    torch_gc()  # 执行GPU内存清理
    return answer  # 返回响应

if __name__ == '__main__':
    # 加载预训练的分词器和模型
    tokenizer = AutoTokenizer.from_pretrained("G:\PengRui\Models\Qwen-7B-Chat\qwen\Qwen-7B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("G:\PengRui\Models\Qwen-7B-Chat\qwen\Qwen-7B-Chat", device_map="auto", trust_remote_code=True).eval()
    model.generation_config = GenerationConfig.from_pretrained("G:\PengRui\Models\Qwen-7B-Chat\qwen\Qwen-7B-Chat", trust_remote_code=True) # 可指定
    model.eval()  # 设置模型为评估模式
    # 启动FastAPI应用
    # 用6006端口可以将autodl的端口映射到本地，从而在本地使用api
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)  # 在指定端口和主机上启动应用
