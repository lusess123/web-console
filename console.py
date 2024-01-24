from flask import Flask, request, jsonify
from threading import Thread
from transformers import pipeline; 

pipeline = pipeline('sentiment-analysis')
import sys

app = Flask(__name__)

console_input = ""

def handle_command_line_input():
    while True:
        input_data = input("请输入命令行数据: ")
        result = pipeline(input_data)
        # print(result)
        result_str = "I said : {label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
        # 在这里处理命令行输入
        print(f"回复: {result_str}")

@app.route('/input', methods=['GET'])
def receive_input():
    # 获取 JSON 数据
    # data = request.json
    input_data = request.args.get('input', default='', type=str)
    # 在这里处理 Web 接口接收的数据
    print(f"Web 数据接收: {input_data}")
    result = pipeline(input_data)
    result_str = "I said : {label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
    print(f"回复: {result_str}")
    return jsonify({"message": "Input received", "input": input_data, "result": result_str})

if __name__ == '__main__':
    # 创建并启动一个线程来处理命令行输入
    command_line_thread = Thread(target=handle_command_line_input)
    command_line_thread.daemon = True  # 设置为守护线程
    command_line_thread.start()

    # 启动 Flask 应用
    app.run(debug=True, port=5000)
