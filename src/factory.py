import logging
from flask import Flask
from src.controller import routes, tasks

logger = logging.getLogger()


def get_celery_config():
    return {
        'CELERY_BROKER_URL': 'redis://:p3b8957d190ff7648c24992a1bef7a17270356a2ca80a21271be22c3de925b349@localhost:6379/0',
        'CELERY_RESULT_BACKEND': 'redis://:p3b8957d190ff7648c24992a1bef7a17270356a2ca80a21271be22c3de925b349@localhost:6379/0'
    }


def create_app():
    return entry_point(mode='app')


def create_worker():
    return entry_point(mode='worker')


def entry_point(mode='app'):
    app = Flask(__name__)
    
    celery_config = get_celery_config()
    app.config['CELERY_BROKER_URL'] = celery_config['CELERY_BROKER_URL']
    app.config['CELERY_RESULT_BACKEND'] = celery_config['CELERY_RESULT_BACKEND']

    configure_celery(app, tasks.celery)

    app.register_blueprint(routes.bp, url_prefix='/api/v1')

    if mode == 'app':
        return app
    elif mode == 'worker':
        return tasks.celery


def configure_celery(app, celery):

    celery.conf.broker_url = app.config['CELERY_BROKER_URL']
    celery.conf.result_backend = app.config['CELERY_RESULT_BACKEND']

    task_base = celery.Task

    class AppContextTask(task_base):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return task_base.__call__(self, *args, **kwargs)
    celery.Task = AppContextTask

    celery.finalize()