from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import traceback

app = Flask(__name__)

# 🔐 Introdueix aquí la teva clau secreta d'OpenAI (NO recomanat en producció)
api_key = "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"

# Inicialització del client OpenAI
client = OpenAI(api_key=api_key)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_document():
    try:
        data = request.json
        print("📥 JSON rebut:", data)

        prompt = data.get('prompt', '')
        print("✏️ Prompt:", prompt)

        if not prompt:
            return jsonify({'error': 'El prompt està buit'}), 400

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ets un assistent que genera documents clars i professionals."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        print("✅ Resposta OpenAI:", response)
        document = response.choices[0].message.content
        return jsonify({'document': document})
    except Exception as e:
        print("❌ ERROR durant la generació:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
