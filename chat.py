"""Local or reverse-proxied WebSocket chat. One worker owns one accelerator."""
import asyncio
import queue
import os
import hashlib
import hmac
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from runtime import ROOT, MODELS, Runtime, Session

app = FastAPI()
if os.environ.get('CHAMBER_DEVICE', '').startswith('cuda'):
    from gpu_runtime import GpuRuntime
    runtime = GpuRuntime()
else:
    runtime = Runtime()

@app.get('/')
async def index():
    return FileResponse(ROOT / 'static/index.html')

@app.get('/api/status')
async def status():
    return runtime.status()

def check_model_password(password: str):
    expected = os.environ.get('CHAMBER_MODEL_PASSWORD_SHA256', '')
    actual = hashlib.sha256(password.encode()).hexdigest()
    if not expected or not hmac.compare_digest(actual, expected):
        raise HTTPException(status_code=403, detail='Неверный пароль')

@app.post('/api/model-unlock')
async def unlock(x_model_password: str = Header(default='')):
    check_model_password(x_model_password)
    return {'ok': True}

@app.post('/api/load/{model}')
async def load(model: str, x_model_password: str = Header(default='')):
    check_model_password(x_model_password)
    try:
        await asyncio.to_thread(runtime.load, model)
        return runtime.status()
    except Exception as exc:
        return JSONResponse({**runtime.status(), 'error': str(exc)}, status_code=409 if isinstance(exc, ValueError) else 500)

@app.websocket('/ws')
async def chat(socket: WebSocket):
    origin = socket.headers.get('origin')
    if origin and urlparse(origin).netloc != socket.headers.get('host'):
        await socket.close(code=1008)
        return
    await socket.accept()
    session = Session()
    history, history_model, task = [], None, None
    async def reply(text):
        nonlocal history
        events = queue.Queue()
        worker = asyncio.create_task(asyncio.to_thread(runtime.generate, session,
            history + [{'role': 'user', 'content': text}], events))
        answer, failed = '', False
        try:
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if event['type'] == 'token':
                    answer += event['text']
                if event['type'] == 'replace':
                    answer = event['text']
                if event['type'] == 'error':
                    failed = True
                if event['type'] == 'done':
                    if not failed and answer:
                        history += [{'role': 'user', 'content': text}, {'role': 'assistant', 'content': answer}]
                    await worker
                    await socket.send_json(event)
                    break
                await socket.send_json(event)
        finally:
            session.stop.set()
            await worker
    try:
        while True:
            data = await socket.receive_json()
            try:
                if not isinstance(data, dict):
                    raise ValueError('Некорректная команда')
                action = data.get('type')
                if action == 'dose':
                    value = session.set_dose(data['value'])
                    await socket.send_json({'type': 'dose', 'value': value})
                elif action == 'stop':
                    session.stop.set()
                elif action == 'reset':
                    if task and not task.done():
                        raise ValueError('Сначала остановите текущий ответ')
                    history, history_model = [], None
                    await socket.send_json({'type': 'reset'})
                elif action == 'message':
                    if task and not task.done():
                        raise ValueError('Ответ ещё генерируется')
                    text = data.get('text', '').strip()
                    key = data.get('model', 'qwen3-4b')
                    if not text:
                        raise ValueError('Введите сообщение')
                    session.set_dose(data.get('dose', 0))
                    runtime.reserve(key)
                    if history_model != key:
                        history = []
                        history_model = key
                    session.stop.clear()
                    try:
                        await socket.send_json({'type': 'started'})
                        task = asyncio.create_task(reply(text))
                    except BaseException:
                        with runtime.lock:
                            runtime.active -= 1
                        raise
                else:
                    raise ValueError('Неизвестная команда')
            except (ValueError, KeyError, TypeError) as exc:
                await socket.send_json({'type': 'error', 'message': str(exc)})
    except WebSocketDisconnect:
        pass
    finally:
        session.stop.set()
        if task:
            await asyncio.gather(task, return_exceptions=True)
