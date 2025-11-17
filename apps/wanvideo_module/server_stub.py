"""
Server Stub for Infinite Talk Pipeline
Provides minimal server interface without genesis dependency

This stub is injected into sys.modules to replace 'server' imports
from ComfyUI custom nodes that would otherwise require genesis.

Author: eddy
Date: 2025-11-17
"""


class PromptQueue:
    """Stub PromptQueue class"""
    
    def __init__(self):
        self.queue = []
    
    def get(self, *args, **kwargs):
        """Stub method - returns empty"""
        return None
    
    def put(self, *args, **kwargs):
        """Stub method - does nothing"""
        pass
    
    def wipe_queue(self, *args, **kwargs):
        """Stub method - does nothing"""
        pass
    
    def get_current_queue(self, *args, **kwargs):
        """Stub method - returns empty list"""
        return ([], [])


class RouterStub:
    """Stub for aiohttp router"""
    
    def add_static(self, prefix, path, **kwargs):
        """Stub method for adding static routes"""
        pass
    
    def add_route(self, method, path, handler, **kwargs):
        """Stub method for adding routes"""
        pass
    
    def add_get(self, path, handler, **kwargs):
        """Stub method for adding GET routes"""
        pass
    
    def add_post(self, path, handler, **kwargs):
        """Stub method for adding POST routes"""
        pass


class AppStub:
    """Stub for aiohttp application"""
    
    def __init__(self):
        self.router = RouterStub()
    
    def add_routes(self, routes):
        """Stub method for adding routes"""
        pass


class RoutesStub:
    """Stub for routes decorator"""
    
    @staticmethod
    def get(path):
        """Stub decorator for GET routes"""
        def decorator(func):
            return func
        return decorator
    
    @staticmethod
    def post(path):
        """Stub decorator for POST routes"""
        def decorator(func):
            return func
        return decorator


class PromptServer:
    """Stub PromptServer class"""
    
    class _Instance:
        """Singleton instance stub"""
        def __init__(self):
            self.client_id = None
            self.loop = None
            self.prompt_queue = PromptQueue()  # Add prompt_queue attribute
            self.number = 0
            self.app = AppStub()  # Add app with router
            self.routes = RoutesStub()  # Add routes decorator
        
        def send_sync(self, *args, **kwargs):
            """Stub method - does nothing"""
            pass
        
        def send(self, *args, **kwargs):
            """Stub method - does nothing"""
            pass
        
        def queue_updated(self, *args, **kwargs):
            """Stub method - does nothing"""
            pass
        
        def add_routes(self, *args, **kwargs):
            """Stub method - does nothing"""
            pass
        
        def get_queue_info(self, *args, **kwargs):
            """Stub method - returns empty queue info"""
            return {"queue_running": [], "queue_pending": []}
    
    # Singleton instance
    instance = _Instance()
    
    @classmethod
    def publish(cls, *args, **kwargs):
        """Stub method - does nothing"""
        pass


# Stub web module
class WebStub:
    """Stub for server.web"""
    
    class RouteDef:
        """Stub route definition"""
        def __init__(self, method, path, handler):
            self.method = method
            self.path = path
            self.handler = handler
    
    class Request:
        """Stub request object"""
        def __init__(self):
            self.rel_url = None
            self.match_info = {}
    
    class Response:
        """Stub response object"""
        def __init__(self, *args, **kwargs):
            pass
    
    @staticmethod
    def get(path):
        """Stub decorator for GET routes"""
        def decorator(func):
            return func
        return decorator
    
    @staticmethod
    def post(path):
        """Stub decorator for POST routes"""
        def decorator(func):
            return func
        return decorator
    
    @staticmethod
    def static(prefix, path):
        """Stub for static routes"""
        pass


# Create web stub instance
web = WebStub()


# Stub routes list
routes = []


def add_routes(route_list):
    """Stub function to add routes"""
    pass


# For compatibility with VideoHelperSuite
class BinaryEventTypes:
    """Stub for binary event types"""
    PREVIEW_IMAGE = 1
    UNENCODED_PREVIEW_IMAGE = 2


def send_sync(event, data, sid=None):
    """Stub function for sending sync messages"""
    pass


def send(event, data, sid=None):
    """Stub function for sending messages"""
    pass


# Stub for folder_paths (sometimes imported together with server)
class FolderPathsStub:
    """Stub for folder_paths"""
    @staticmethod
    def get_directory_by_type(type_name):
        return ""
    
    @staticmethod
    def get_filename_list(folder_name):
        return []


def __getattr__(name):
    """
    Fallback for any other attributes
    Returns a stub function that does nothing
    """
    def stub(*args, **kwargs):
        pass
    return stub
