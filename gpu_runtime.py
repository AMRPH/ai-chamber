"""Supervisor for an isolated, continuously batched CUDA engine."""
import json
import queue
import subprocess
import sys
import threading
import uuid
from runtime import ROOT, MODELS, Runtime

class GpuRuntime(Runtime):
    def __init__(self):
        super().__init__()
        self.process = None
        self.queues = {}
        self.send_lock = threading.Lock()
        self.ready_queue = queue.Queue()
        self.log = None

    def send(self, message):
        with self.send_lock:
            self.process.stdin.write(json.dumps(message,ensure_ascii=False)+'\n')
            self.process.stdin.flush()

    def load(self,key):
        if key not in MODELS:
            raise ValueError('Неизвестная модель')
        with self.lock:
            if self.state=='ready' and self.key==key:
                return
            if self.active:
                raise ValueError('Смена модели доступна после завершения активных ответов')
            if self.state=='loading':
                raise ValueError('Модель ещё загружается')
            self.state,self.error,self.key='loading',None,key
        try:
            if self.process:
                if self.process.poll() is None:
                    self.send({'type':'shutdown'})
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=30)
                self.log.close()
            self.ready_queue=queue.Queue()
            self.log=(ROOT/'cuda-engine.log').open('a')
            self.process=subprocess.Popen([sys.executable,'-u',str(ROOT/'cuda_engine.py'),key],
                                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                                          text=True,bufsize=1,cwd=ROOT)
            process=self.process
            ready_queue=self.ready_queue
            def read():
                try:
                    for line in process.stdout:
                        event=json.loads(line)
                        if event['type'] in ('ready','fatal'):
                            ready_queue.put(event)
                        if event.get('id') in self.queues:
                            self.queues[event['id']].put(event)
                finally:
                    ready_queue.put({'type':'fatal','message':'CUDA engine завершился; см. cuda-engine.log'})
                    with self.lock:
                        if process is self.process and self.state != 'loading':
                            self.state='error'
                            for events in self.queues.values():
                                events.put({'type':'error','message':'CUDA engine завершился'})
                                events.put({'type':'done','stopped':False})
            threading.Thread(target=read,daemon=True).start()
            event=ready_queue.get()
            if event['type']=='fatal':
                raise RuntimeError(event['message'])
            with self.lock:
                self.device,self.state=event['device'],'ready'
        except Exception as exc:
            with self.lock:
                self.state,self.error='error',str(exc)
            raise

    def generate(self,session,messages,events):
        rid=uuid.uuid4().hex
        incoming=queue.Queue()
        with self.lock:
            self.queues[rid]=incoming
        try:
            with session.lock:
                dose=session.dose
            self.send({'type':'generate','id':rid,'messages':messages,'dose':dose})
            stopped=False
            while True:
                with session.lock:
                    current=session.dose
                if current!=dose:
                    self.send({'type':'dose','id':rid,'value':current})
                    dose=current
                if session.stop.is_set() and not stopped:
                    self.send({'type':'stop','id':rid})
                    stopped=True
                try:
                    event=incoming.get(timeout=0.02)
                except queue.Empty:
                    continue
                events.put(event)
                if event['type']=='done':
                    return
        except Exception as exc:
            events.put({'type':'error','message':str(exc)})
            events.put({'type':'done','stopped':session.stop.is_set()})
        finally:
            with self.lock:
                self.queues.pop(rid,None)
                self.active-=1
