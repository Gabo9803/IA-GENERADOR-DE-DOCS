from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import openai
import os
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO
from datetime import datetime
import re
import json

app = Flask(__name__)

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY no está configurada en las variables de entorno")

# Configurar cliente OpenAI
client = openai.OpenAI(api_key=openai_api_key)

# Cache para respuestas y estilos
response_cache = {}
style_profiles = {}

def parse_markdown_to_reportlab(text, style_profile=None):
    """Convierte Markdown a elementos de reportlab con soporte para estilos personalizados."""
    styles = getSampleStyleSheet()
    font_name = style_profile.get('font_name', 'Helvetica') if style_profile else 'Helvetica'
    font_size = style_profile.get('font_size', 12) if style_profile else 12
    body_style = ParagraphStyle(
        name='Body', fontSize=font_size, leading=font_size * 1.33, spaceAfter=8, fontName=font_name
    )
    bold_style = ParagraphStyle(
        name='Bold', fontSize=font_size, leading=font_size * 1.33, spaceAfter=8, fontName=f'{font_name}-Bold'
    )
    elements = []

    # Soporte para tablas Markdown
    table_pattern = re.compile(r'\|(.+?)\|\n\|[-|:\s]+\|\n((?:\|.+?\|(?:\n|$))+)')
    text_parts = table_pattern.split(text)
    
    for i, part in enumerate(text_parts):
        if i % 3 == 2:  # Contenido de la tabla
            rows = [row.strip().split('|')[1:-1] for row in part.strip().split('\n')]
            table_data = [[cell.strip() for cell in row] for row in rows]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), font_size),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))
        elif i % 3 != 1:  # Texto normal
            lines = part.split('\n')
            for line in lines:
                line = line.replace('\n', '<br />')
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                if line.strip().startswith('- '):
                    line = f'• {line[2:]}'
                    elements.append(Paragraph(line, body_style))
                else:
                    elements.append(Paragraph(line, bold_style if '**' in line else body_style))
                elements.append(Spacer(1, 0.1 * inch))
    
    return elements

def analyze_document(file):
    """Analiza un documento subido para extraer estilo, formato y tipografía."""
    style_profile = {
        'font_name': 'Helvetica',
        'font_size': 12,
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'tone': 'neutral',
        'structure': []
    }
    
    content = ""
    if file.filename.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                content += page.extract_text() or ""
                # Análisis de tipografía (aproximado)
                chars = page.chars
                if chars:
                    style_profile['font_name'] = chars[0].get('fontname', 'Helvetica').split('-')[0]
                    style_profile['font_size'] = round(chars[0].get('size', 12))
                # Análisis de márgenes (aproximado)
                style_profile['margins'] = {
                    'top': page.bbox[3] - page.chars[-1]['y1'] if page.chars else 1,
                    'bottom': page.chars[0]['y0'] - page.bbox[1] if page.chars else 1,
                    'left': page.chars[0]['x0'] - page.bbox[0] if page.chars else 1,
                    'right': page.bbox[2] - page.chars[-1]['x1'] if page.chars else 1
                }
    elif file.filename.endswith('.txt'):
        content = file.read().decode('utf-8')
    
    # Análisis de tono y estructura con OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Analiza el tono (formal, informal, técnico) y la estructura (encabezados, párrafos, listas, tablas) del siguiente texto. Devuelve un JSON con 'tone' y 'structure'."},
                {"role": "user", "content": content[:4000]}  # Limitar para evitar tokens excesivos
            ],
            max_tokens=500
        )
        analysis = json.loads(response.choices[0].message.content)
        style_profile['tone'] = analysis.get('tone', 'neutral')
        style_profile['structure'] = analysis.get('structure', [])
    except Exception as e:
        print(f"Error en análisis de tono: {e}")
    
    return style_profile, content

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_document():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        session_id = data.get('session_id', 'default')
        doc_type = data.get('doc_type', 'general')
        tone = data.get('tone', 'neutral')
        length = data.get('length', 'medium')
        language = data.get('language', 'es')
        style_profile_id = data.get('style_profile_id', None)
        
        if not prompt:
            return jsonify({'error': 'El prompt está vacío'}), 400

        # Verificar cache
        cache_key = f"{session_id}:{prompt}:{doc_type}:{tone}:{length}:{language}:{style_profile_id}"
        if cache_key in response_cache:
            return jsonify({'document': response_cache[cache_key]})

        # Construir prompt avanzado
        system_prompt = f"""
        Eres un asistente que genera documentos profesionales en formato Markdown. 
        - Tipo de documento: {doc_type} (e.g., carta formal, informe, correo, contrato, currículum).
        - Tono: {tone} (formal, informal, técnico).
        - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
        - Idioma: {language} (e.g., es para español, en para inglés, fr para francés).
        - Usa encabezados (#, ##), listas (-), negritas (**), y tablas (|...|) cuando sea apropiado.
        - Si se proporciona un estilo, síguelo: {style_profiles.get(style_profile_id, {}) if style_profile_id else 'ninguno'}.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        document = response.choices[0].message.content
        response_cache[cache_key] = document
        return jsonify({'document': document})
    except openai.AuthenticationError:
        return jsonify({'error': 'Error de autenticación con OpenAI. Verifica la clave API.'}), 401
    except openai.RateLimitError:
        return jsonify({'error': 'Límite de solicitudes alcanzado. Intenta de nuevo más tarde.'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    try:
        data = request.json
        content = data.get('content', '')
        style_profile_id = data.get('style_profile_id', None)
        if not content:
            return jsonify({'error': 'El contenido está vacío'}), 400

        # Crear buffer para el PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch, 
            topMargin=inch, 
            bottomMargin=0.75*inch
        )
        
        # Estilos
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, spaceAfter=12, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle(name='Subtitle', fontSize=10, leading=14, spaceAfter=10, fontName='Helvetica-Oblique')

        # Elementos del PDF
        story = []
        story.append(Paragraph("Documento Generado por IA", title_style))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Aplicar estilo personalizado si existe
        style_profile = style_profiles.get(style_profile_id, None)
        story.extend(parse_markdown_to_reportlab(content, style_profile))
        
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

@app.route('/analyze_document', methods=['POST'])
def analyze_document_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file']
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.txt')):
            return jsonify({'error': 'Formato no soportado. Usa PDF o TXT.'}), 400
        
        style_profile, content = analyze_document(file)
        profile_id = str(len(style_profiles))
        style_profiles[profile_id] = style_profile
        
        return jsonify({
            'style_profile_id': profile_id,
            'style_profile': style_profile,
            'content_summary': content[:500]  # Resumen para evitar respuestas largas
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/style_profiles', methods=['GET'])
def get_style_profiles():
    return jsonify(style_profiles)

@app.route('/style_profiles/<profile_id>', methods=['DELETE'])
def delete_style_profile(profile_id):
    if profile_id in style_profiles:
        del style_profiles[profile_id]
        return jsonify({'message': 'Perfil de estilo eliminado'})
    return jsonify({'error': 'Perfil no encontrado'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))