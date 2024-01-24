from transformers import pipeline;
from flask import Flask, request, jsonify

pipeline = pipeline('sentiment-analysis')

app = Flask(__name__)

@app.route('/interact', methods=['GET'])
def interact_with_cli():
    if request.method == 'POST':
        # 对于 POST 请求，也可以从查询字符串获取数据
        query_param = request.args.get('param', default='', type=str)
        return jsonify({'message': 'POST 请求已接收', 'query_param': query_param})
    else:
        # 从查询字符串中获取参数
        query_param = request.args.get('param', default='', type=str)
        result = pipeline(query_param)
        result_str = "{label}（{score:.3f}）".format(label=result[0]['label'].lower(), score=result[0]['score'])
        
        
        with open('output.txt', 'r') as file:
              content = file.read()

        return jsonify({'input': query_param, 'result': result_str, 'output': content })

if __name__ == '__main__':
    app.run(debug=True)
