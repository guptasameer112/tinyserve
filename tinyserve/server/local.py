"""Start and stop Uvicorn on an OS-assigned loopback port inside a notebook."""
import asyncio
import socket
from contextlib import asynccontextmanager

import uvicorn


@asynccontextmanager
async def serve_local(app):
    """No public bind, tunnel, subprocess, reload or extra model-loading worker."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", log_level="warning",
                                         access_log=False, lifespan="on"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = listener.getsockname()[1]
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async def ready():
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("local server exited before becoming ready")
                    await asyncio.sleep(0.01)
            await asyncio.wait_for(ready(), timeout=30)
            yield "http://127.0.0.1:%d" % port
        finally:
            server.should_exit = True
            try:
                await asyncio.wait_for(task, timeout=10)
            finally:
                if not task.done():
                    task.cancel()
