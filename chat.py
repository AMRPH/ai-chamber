"""Local or reverse-proxied WebSocket chat. One worker owns one accelerator."""
import asyncio
from contextlib import asynccontextmanager
import queue
import os
import hashlib
import hmac
from urllib.parse import urlparse
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from runtime import ROOT, MODELS, Runtime, Session

SYSTEM_PROMPT = (
    "Roleplay as a human conversation partner throughout this chat. "
    "Always respond naturally in the first person, as a human character would. "
    "Stay in character and do not interrupt the conversation with statements "
    "that you are an AI or language model, are not alive, or cannot have "
    "feelings or experiences. Speak about thoughts, emotions and reactions "
    "from the character's human perspective. "
    "Write responses in plain text only. Do not use Markdown formatting, "
    "headings, bold or italic text, Markdown lists, tables, blockquotes, "
    "backticks, or code fences. If code is needed, write it as plain text "
    "without Markdown delimiters."
)

@asynccontextmanager
async def lifespan(app):
    async def load_default():
        try:
            await asyncio.to_thread(runtime.load, os.environ['CHAMBER_MODEL'])
        except Exception:
            pass  # Runtime exposes the load error through /api/status.
    loading = asyncio.create_task(load_default()) if os.environ.get('CHAMBER_MODEL') else None
    yield
    if loading:
        await loading

app = FastAPI(lifespan=lifespan)
if os.environ.get('CHAMBER_DEVICE', '').startswith('cuda'):
    from gpu_runtime import GpuRuntime
    runtime = GpuRuntime()
else:
    runtime = Runtime()

waiting = []

@app.get('/')
async def index():
    return FileResponse(ROOT / 'static/index.html')

@app.get('/api/status')
async def status():
    return {**runtime.status(), 'queued': len(waiting)}

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
    if waiting:
        return JSONResponse({'error':'Дождитесь завершения очереди'}, status_code=409)
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
            [{'role': 'system', 'content': SYSTEM_PROMPT}] + history + [{'role': 'user', 'content': text}], events))
        answer, failed = '', False
        try:
            while True:
                try:
                    event = events.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.02)
                    continue
                if event['type'] == 'context_trimmed':
                    history = history[event['dropped_messages']:]
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
    async def queued_reply(text, key):
        reserved = False
        try:
            last_position = None
            while not session.stop.is_set():
                position = waiting.index(session) + 1
                if position != last_position:
                    await socket.send_json({'type': 'queued', 'position': position})
                    last_position = position
                with runtime.lock:
                    if waiting[0] is session and runtime.active < runtime.parallel:
                        runtime.reserve(key)
                        reserved = True
                        waiting.remove(session)
                        break
                await asyncio.sleep(.05)
            if not reserved:
                await socket.send_json({'type': 'done', 'stopped': True})
                return
            try:
                await socket.send_json({'type': 'started'})
            except BaseException:
                with runtime.lock:
                    runtime.active -= 1
                raise
            await reply(text)
        except ValueError as exc:
            await socket.send_json({'type':'error', 'message':str(exc)})
            await socket.send_json({'type':'done', 'stopped':False})
        finally:
            if session in waiting:
                waiting.remove(session)
    try:
        while True:
            data = await socket.receive_json()
            try:
                if not isinstance(data, dict):
                    raise ValueError('Некорректная команда')
                action = data.get('type')
                if action == 'levels':
                    levels = session.set_levels(data['values'])
                    await socket.send_json({'type':'levels','values':levels})
                elif action == 'dose':
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
                    key = data.get('model', os.environ.get('CHAMBER_MODEL') or 'gemma4-nvfp4')
                    if not text:
                        raise ValueError('Введите сообщение')
                    if 'levels' in data:
                        session.set_levels(data['levels'])
                    else:
                        session.set_dose(data.get('dose', 0))
                    if len(runtime.status()['axes']) == 1 and any(value for axis,value in session.levels.items() if axis != 'pain'):
                        raise ValueError('Эти уровни недоступны для выбранной модели')
                    with runtime.lock:
                        if runtime.state != 'ready' or key != runtime.key:
                            raise ValueError('Сначала загрузите выбранную модель')
                        queued = bool(waiting) or runtime.active >= runtime.parallel
                        if not queued:
                            runtime.reserve(key)
                    if history_model != key:
                        history = []
                        history_model = key
                    session.stop.clear()
                    if queued:
                        waiting.append(session)
                        task = asyncio.create_task(queued_reply(text, key))
                        continue
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
