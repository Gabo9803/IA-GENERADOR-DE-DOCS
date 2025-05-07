from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import openai
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from datetime import datetime
import re

app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Cache para respuestas frecuentes
response_cache = {}

def parse_markdown_to_reportlab(text):
    """Convierte Markdown simple a elementos de reportlab."""
    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(name='Body', fontSize=12, leading=16, spaceAfter=8, fontName='Helvetica')
    bold_style = ParagraphStyle(name='Bold', fontSize=12, leading=16, spaceAfter=8, fontName='Helvetica-Bold')
    elements = []
    
    lines = text.split('\n')
    for line in lines:
        line = line.replace('\n', '<br />')
        # Soporte básico para negritas (**texto**)
        line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
        # Soporte para listas (- texto)
        if line.strip().startswith('- '):
            line = f'• {line[2:]}'
            elements.append(Paragraph(line, body_style))
        else:
            elements.append(Paragraph(line, bold_style if '**' in line else body_style))
        elements.append(Spacer(1, 0.1 * inch))
    
    return elements

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_document():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        if not prompt:
            return jsonify({'error': 'El prompt está vacío'}), 400

        # Verificar cache
        cache_key = f"{session_id}:{prompt}"
        if cache_key in response_cache:
            return jsonify({'document': response_cache[cache_key]})

        # Llamada a la API de OpenAI
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente que genera documentos claros y profesionales en formato Markdown. Usa listas, negritas y encabezados cuando sea apropiado."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        document = response.choices[0].message.content
        response_cache[cache_key] = document
        return jsonify({'document': document})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        content = data.get('content', '')
        if not content:
            return jsonify({'error': 'El contenido está vacío'}), 400

        # Crear buffer para el PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=0.75*inch, leftMargin=0.75*inch, topMargin=inch, bottomMargin=0.75*inch)
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, spaceAfter=12, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle(name='Subtitle', fontSize=10, leading=14, spaceAfter=10, fontName='Helvetica-Oblique')

        # Elementos del PDF
        story = []
        story.append(Paragraph("Documento Generado por IA", title_style))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Parsear Markdown
        story.extend(parse_markdown_to_reportlab(content))
        
        doc.build(story)
        buffer.seek(0)
        return send_file(
            buffer,
            as_attachment=True,
            download_name='documento_generado.pdf',
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))