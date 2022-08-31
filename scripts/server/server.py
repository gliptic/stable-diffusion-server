import torch
import msgpack
import asyncio, socket
import time
import random
import io, os, json, re, hashlib
import qoi
import fpng_py
from threading import Thread
from PIL import Image
from threading import Lock
from contextlib import suppress

def take_job_from_request(request, batch):
    batch.append({
        'id': request['id'],
        'prompt': request["prompt"],
        "seed": request["seed"],
        'scale': request["scale"],
        'steps': request["steps"],
        'w': request["w"],
        'h': request["h"],
        'index': request['index'],
        'client': request['client'],
        'sampler': request['sampler'],
    })
    request["seed"] = (request["seed"] + 1) & 0x7fffffff
    request["count"] -= 1
    request['index'] += 1

def key_for_request(request):
    return f"{request['w']}_{request['h']}_{request['scale']}_{request['steps']}_{request['sampler']}"

class Client:
    def __init__(self, socket, server):
        self.socket = socket
        self.unpacker = msgpack.Unpacker()
        self.server = server
        self.connected = True

    async def handle_client(self):
        loop = asyncio.get_event_loop()
        request = None
        try:
            while True:
                print('Waiting for data from client...')
                request = (await loop.sock_recv(self.socket, 255))
                print(f'Got data from client {request}')

                if len(request) == 0:
                    print('Got empty')
                    return

                self.unpacker.feed(request)
                for msg in self.unpacker:
                    self.handle_message(msg)
                #response = str(eval(request)) + '\n'
                #await loop.sock_sendall(self.client, response.encode('utf8'))
        finally:
            self.connected = False
            self.socket.close()

    def send_response(self, buffer):
        if self.connected:
            print(f'sending {len(buffer)} bytes')
            self.socket.send(len(buffer).to_bytes(4, 'little'))
            self.socket.send(buffer)
        else:
            print("client no longer connected, cannot send result")

    def handle_message(self, msg):
        print('Got message', msg)
        self.server.handle_message(msg, self)

class Event_ts(asyncio.Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

    def set(self):
        self._loop.call_soon_threadsafe(super().set)

    def clear(self):
        self._loop.call_soon_threadsafe(super().clear)

class Server:
    def __init__(self):
        self.queue = []
        self.bucket_mutex = Lock()
        self.counts_in_dirs = {}

    def handle_message(self, msg, client):
        if msg['ty'] == "request":
            request = msg['request']
            request['client'] = client
            self.add_request(request)

    def add_request(self, request):
        request['index'] = 0
        if request["seed"] < 0:
            request["seed"] = random.randint(0, 0x7fffffff)

        with self.bucket_mutex:
            self.queue.append(request)

    def steps_for_batch(self, batch):
        return batch[0]['steps']

    def scale_for_batch(self, batch):
        return batch[0]['scale']

    def prompts_for_batch(self, batch):
        return [job['prompt'] for job in batch]

    def sampler_for_batch(self, batch):
        return batch[0]['sampler']

    def shape_for_batch(self, batch):
        C = 4
        f = 8
        return [C, batch[0]['h'] // f, batch[0]['w'] // f]

    def startcodes_for_batch(self, batch):
        C = 4
        f = 8
        start_codes = []
        for job in batch:
            generator = torch.Generator()
            generator.manual_seed(job["seed"])
            start_codes.append(torch.randn([C, job['h'] // f, job['w'] // f], generator=generator))
        return torch.stack(start_codes, 0)

    def encode_image(self, job, image):
        start_time = time.time()
        png_bytes = fpng_py.fpng_encode_image_to_memory(image.tobytes(), job['w'], job['h'], 3, fpng_py.CompressionFlags.FPNG_ENCODE_SLOWER)
        end_time = time.time()
        png_size = len(png_bytes)
        print(f"fpng: {end_time - start_time}, size: {png_size}")

        return png_bytes

    def save_image(self, job, image, outpath):
        hashkey = json.dumps({ 'prompt': job['prompt'] })
        hash = hashlib.sha1(hashkey.encode('utf8')).hexdigest()
        dirname = re.sub(r"[^A-Za-z0-9,]", lambda _: "_", job['prompt'])[:140] + hash

        sample_path = os.path.join(outpath, "samples", dirname)

        if not os.path.isdir(sample_path):
            os.makedirs(sample_path, exist_ok=True)

            with open(os.path.join(sample_path, 'options.txt'), 'a') as f:
                json.dump({ 'prompt': job['prompt']}, f)

        if sample_path not in self.counts_in_dirs:
            self.counts_in_dirs[sample_path] = len(os.listdir(sample_path)) - 1

        start_time = time.time()
        png_bytes = fpng_py.fpng_encode_image_to_memory(image.tobytes(), job['w'], job['h'], 3, fpng_py.CompressionFlags.FPNG_ENCODE_SLOWER)
        end_time = time.time()
        png_size = len(png_bytes)
        print(f"fpng: {end_time - start_time}, size: {png_size}")

        sampler = job['sampler'].replace('_', '-')
        with open(os.path.join(sample_path, f"{self.counts_in_dirs[sample_path]:05}_{job['seed']}_{job['scale']}_{job['steps']}_{sampler}.png"), 'wb') as f:
            f.write(png_bytes)

        self.counts_in_dirs[sample_path] += 1

        return png_bytes

    def send_response(self, job, image_bytes):
        response = {
            'ty': 'result',
            'result': {
                'id': job['id'],
                'index': job['index'],
                "seed": job["seed"],
                'scale': job['scale'],
                'steps': job['steps'],
                'prompt': job['prompt'],
                'sampler': job['sampler'],
                'image': image_bytes
            }
        }

        b = msgpack.packb(response)
        job['client'].send_response(b)

    def batch_requests(self, max_count):
        batch = []
        with self.bucket_mutex:
            if len(self.queue) > 0:
                index = 0
                batch_key = key_for_request(self.queue[0])

                while index < len(self.queue) and len(batch) < max_count:
                    request = self.queue[index]
                    if batch_key == key_for_request(request):
                        take_job_from_request(request, batch)

                        # Delete or move last
                        del self.queue[index]
                        if request['count'] > 0:
                            self.queue.append(request)
                    else:
                        index += 1

        return batch

    def wait_for_requests(self):
        print('Wait for requests')
        while True:
            with self.bucket_mutex:
                if len(self.queue) > 0:
                    time.sleep(0.2) # Allow any straggling requests to arrive
                    return True
            time.sleep(0.1)

    def start(self):
        def start_serv():
            print("Started server thread")
            asyncio.run(self.run())

        serv_thread = Thread(target=start_serv)
        serv_thread.start()

    def stop(self):
        pass
        #self.stop_event.set()

    # async def wait_for_stop(self):
    #     await self.stop_event.wait()

    #     print('Stop event triggered')

    #     loop = asyncio.get_event_loop()
    #     pending = asyncio.all_tasks()
    #     for task in pending:
    #         task.cancel()
    #         # Now we should await task to execute it's cancellation.
    #         # Cancelled task raises asyncio.CancelledError that we can suppress:
    #         with suppress(asyncio.CancelledError):
    #             loop.run_until_complete(task)

    async def run(self):
        #self.stop_event = Event_ts()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('localhost', 9999))
        print('Listening...')
        self.socket.listen(8)
        self.socket.setblocking(False)

        loop = asyncio.get_event_loop()

        # loop.create_task(self.wait_for_stop())

        while True:
            print('Waiting for client...')

            client, _ = await loop.sock_accept(self.socket)
            print('Accepted client')
            
            loop.create_task(Client(client, self).handle_client())

    