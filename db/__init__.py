# from . import websocket
from .tracker import tasktrack

# This will make sure the app is always imported when
# Django starts so that shared_task will use this app.
from .celery_base import app as celery_app

__all__ = ('celery_app',)