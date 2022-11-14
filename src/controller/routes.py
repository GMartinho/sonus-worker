from flask import jsonify, Blueprint, request
from src.controller import tasks
import logging
import uuid


bp = Blueprint('tasks', __name__)
logger = logging.getLogger()


@bp.route('/midi-transcriber', methods=['POST'])
def midi_transcriber():
    record_key = request.json['record_key']
    record_url = request.json['record_url']
    task = tasks.midi_record_generator.delay(record_key, record_url)

    return jsonify({
        'success': True,
        'token': task.id,
    }), 202

@bp.route('/midi-transcriber/<token>', methods=['GET'])
def get_status(token):
    state = tasks.check_status(token)
    return jsonify({
        'status': state.get('status'),
        'midi-record': state.get('info'),
    }), 200