from flask import Flask, request, jsonify
from threading import Thread
from transformers import pipeline; 
from datetime import datetime
from flask_cors import CORS
from flask_socketio import SocketIO, emit

import logging

pipeline = pipeline('sentiment-analysis',model="distilbert-base-uncased-finetuned-sst-2-english")
import sys

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.logger.disabled = True

# 关闭Werkzeug的日志
log = logging.getLogger('werkzeug')
log.disabled = True

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins='*')



console_input = ""
# global indexNum 
indexNum = 0.0
hostoryMessage = []

def handle_command_line_input():
    while True:
        input_data = input("请输入命令行数据: ")
        item1, item2 = consoleInput(input_data)
        # with app.app_context():
        socketio.emit('append', {'message': [item1 , item2]}, namespace='/chat')

def consoleInput(input_data):
    global indexNum 
    global hostoryMessage
    global app 
    indexNum = indexNum + 1
    item1 = [indexNum, input_data, "user", datetime.now().timestamp()]
    hostoryMessage.append(item1)
    result = pipeline(input_data)
        # print(result)
    result_str = "{label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
        # 在这里处理命令行输入
    indexNum = indexNum + 1
    item2 = [indexNum, result_str, "ai", datetime.now().timestamp()]
    hostoryMessage.append(item2)
    print(f"回复: {result_str}")
    return item1,item2

@app.route('/input', methods=['GET'])
def receive_input():
    global indexNum 
    global hostoryMessage
    # 获取 JSON 数据
    # data = request.json
    input_data = request.args.get('input', default='', type=str)
    if(input_data == ''):
        return jsonify({"history": hostoryMessage})
    indexNum = indexNum + 1
    # 在这里处理 Web 接口接收的数据
    hostoryMessage.append([indexNum, input_data, "user", datetime.now().timestamp()])
    print(f"Web 数据接收: {input_data}")
    result = pipeline(input_data)
    result_str = "{label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
    print(f"{result_str}")
    indexNum = indexNum + 1
    hostoryMessage.append([indexNum, result_str, "ai", datetime.now().timestamp()])
    resJson =  jsonify({"message": "Input received", "input": input_data, "result": result_str, "history": hostoryMessage})
    return resJson

@socketio.on('connect', namespace='/chat')
def handle_connect():
    item = addMessage('Welcome,You have entered the chat room', "system")
    socketio.emit('append', {'message':[item]}, namespace='/chat')
    print('Client connected')

def addMessage(text, textType):
    global indexNum
    indexNum = indexNum + 1
    item = [indexNum, text, textType, datetime.now().timestamp()]
    return item


@socketio.on('message', namespace='/chat')
def handle_message(text):
    global hostoryMessage
    item = addMessage(text, "user")
    # 在这里处理 Web 接口接收的数据
    hostoryMessage.append(item)
    print(f"Web 数据接收: {text}")

    result = pipeline(text)
    result_str = "{label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
    print(f"{result_str}")

    item1 = addMessage(result_str, "ai")
    hostoryMessage.append(item1)
    
    socketio.emit('append', {'message':[item, item1]}, namespace='/chat')

@socketio.on('disconnect', namespace='/chat')
def handle_disconnect():
    item = addMessage('Goodbye,You have left the chat room', "system")
    socketio.emit('append', {'message':[item]}, namespace='/chat')
    print('Client disconnected')

if __name__ == '__main__':
    # 创建并启动一个线程来处理命令行输入
    command_line_thread = Thread(target=handle_command_line_input)
    command_line_thread.daemon = True  # 设置为守护线程
    command_line_thread.start()

    # 启动 Flask 应用
    app.run(port=5000)
