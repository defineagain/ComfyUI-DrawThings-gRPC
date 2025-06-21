from aiohttp import web
from server import PromptServer

from .. import cancel_request, settings
from .draw_things import get_files

routes = PromptServer.instance.routes


@routes.post('/dt_grpc_files_info')
async def handle_files_info_request(request):
    """
    Returns a list of all files on the Draw Things gRPC server.
    """
    try:
        post = await request.post()
        server = post.get('server')
        port = post.get('port')
        use_tls = post.get('use_tls')

        if server is None or port is None:
            return web.json_response({"error": "Missing server or port parameter"}, status=400)
        all_files = get_files(server, port, use_tls)
        return web.json_response(all_files)
    except Exception as e:
        print(e)
        return web.json_response({"error": "Could not connect to Draw Things gRPC server. Please check the server address and port."}, status=500)


@routes.post('/dt_grpc_preview')
async def handle_preview_request(request):
    """
    Toggles the preview mode on or off.
    """
    try:
        post = await request.post()
        settings.show_preview = False if post.get('preview') == "none" else True
        return web.json_response()
    except Exception as e:
        print(e)
        return web.json_response()


@routes.post('/dt_grpc_interrupt')
async def handle_interrupt_request(request):
    """
    Handles interrupt requests to the gRPC server by setting the cancel request flag.
    """
    cancel_request.cancel()
    return web.json_response()
