import msgpack
import asyncio, socket
import time
import random
from threading import Lock
from contextlib import suppress

class Bucket:
    def __init__(self, key):
        self.key = key
        self.count = 0
        self.requests = []

    def fill_batch(self, max_count, batch):
        while self.count > 0 and len(batch) < max_count:
            request = self.requests[0]
            number_to_add = min(max_count - len(batch), request["count"])
            print(f'Adding {number_to_add} requests from bucket')
            for i in range(number_to_add):
                batch.append({
                    'id': request['id'],
                    'prompt': request["prompt"],
                    'seed': request["seed"],
                    'scale': request["scale"],
                    'steps': request["steps"],
                    'w': request["w"],
                    'h': request["h"],
                    'index': request['index'],
                    'client': request['client'],
                })
                request["seed"] = (request["seed"] + 1) & 0x7fffffff
                request["count"] -= 1
                request['index'] += 1

            if request["count"] == 0:
                del self.requests[0]
            self.count -= number_to_add

    def add_request(self, request):
        request['index'] = 0
        if request['seed'] < 0:
            request['seed'] = random.randint(0, 0x7fffffff)
        self.count += request['count']
        self.requests.append(request)

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
        #self.unpacker = msgpack.Unpacker()
        self.buckets = {}
        self.bucket_mutex = Lock()

    def handle_message(self, msg, client):
        if msg['ty'] == "request":
            request = msg['request']
            request['client'] = client
            self.add_request(request)

    def add_request(self, request):
        bucket_key = f"{request['w']}_{request['h']}_{request['scale']}_{request['steps']}"
        print(f'bucket key {bucket_key}')

        with self.bucket_mutex:
            if bucket_key not in self.buckets:
                self.buckets[bucket_key] = Bucket(bucket_key)
            self.buckets[bucket_key].add_request(request)

    def send_response(self, response, client):
        b = msgpack.packb(response)
        client.send_response(b)

    def batch_requests(self, max_count):
        batch = []
        with self.bucket_mutex:
            bucket = max(self.buckets.values(), key=lambda x: x.count)
            bucket.fill_batch(max_count, batch)
            if bucket.count == 0:
                del self.buckets[bucket.key]
            return batch

    def wait_for_requests(self):
        print('Wait for requests')
        while True:
            with self.bucket_mutex:
                if len(self.buckets) > 0:
                    return True
            time.sleep(0.1)

    def stop(self):
        self.stop_event.set()

    async def wait_for_stop(self):
        await self.stop_event.wait()

        print('Stop event triggered')

        loop = asyncio.get_event_loop()
        pending = asyncio.all_tasks()
        for task in pending:
            task.cancel()
            # Now we should await task to execute it's cancellation.
            # Cancelled task raises asyncio.CancelledError that we can suppress:
            with suppress(asyncio.CancelledError):
                loop.run_until_complete(task)

    async def run(self):
        self.stop_event = Event_ts()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(('localhost', 9999))
        print('Listening...')
        self.socket.listen(8)
        self.socket.setblocking(False)

        loop = asyncio.get_event_loop()

        loop.create_task(self.wait_for_stop())

        while True:
            print('Waiting for client...')

            client, _ = await loop.sock_accept(self.socket)
            print('Accepted client')
            
            loop.create_task(Client(client, self).handle_client())

    