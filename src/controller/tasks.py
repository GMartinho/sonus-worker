from dotenv import load_dotenv
from celery import Celery
import logging
import os
import requests
from src.service import midi_transcriber
from src.service.aws_s3 import upload

load_dotenv()

logger = logging.getLogger()
celery = Celery(__name__, autofinalize=False)

@celery.task(bind=True)
def midi_record_generator(self, record_key, record_url):
    try:
        logger.info("Starting MIDI generation...")
        self.update_state(state='IN_PROGRESS', meta=None)
        record = requests.get(record_url)
        record_parts = record_key.split(':')
        record_filename = record_parts[1]
        midi_filename = record_filename.split('.')[0] + '.midi'
        midi_key = f'{record_parts[0]}:{midi_filename}'

        with open(record_filename, 'wb') as f:
            f.write(record.content)

        midi_transcriber.transcribe(record_filename, midi_filename)

        upload(midi_filename, os.getenv('BUCKET_COMPOSITION_MIDI'), midi_key)

        if os.path.isfile(record_filename):
            os.remove(record_filename)
            print("Record has been deleted")
        else:
            print("Record does not exist") 

        logger.info("Completed MIDI generation...")

        return midi_key
    except NameError:
        print("Something went wrong")
        print(NameError)
    finally:
        if os.path.isfile(midi_filename):
            os.remove(midi_filename)
            print("MIDI has been deleted")
        else:
            print("MIDI does not exist") 


def check_status(task_id):
    task = celery.AsyncResult(task_id)
    state = {'status': task.status, 'info': task.info}
    return state