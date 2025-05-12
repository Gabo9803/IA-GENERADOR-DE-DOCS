from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash, Response
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from dotenv import load_dotenv
import openai
import os
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
from datetime import datetime
import re
import json
import sqlite3
import bcrypt
from collections import Counter
import statistics
import hashlib
import markdown2
from docx import Document
import pytesseract
from PIL import Image

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Cargar variables de entorno
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY no está configurada")

# Configurar cliente OpenAI
client = openai.OpenAI(api_key=openai_api_key)

# Configurar Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Modelo de usuario
class User(UserMixin):
    def __init__(self, id, email):
        self.id = id
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    if user:
        return User(user[0], user[1])
    return None

# Versión del esquema
CURRENT_SCHEMA_VERSION = 3

def init_db():
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS schema_version (
            id INTEGER PRIMARY KEY,
            version INTEGER NOT NULL
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS style_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT,
            fonts TEXT,
            font_size INTEGER,
            tone TEXT,
            margins TEXT,
            structure TEXT,
            text_color TEXT,
            background_color TEXT,
            secondary_colors TEXT,
            alignment TEXT,
            line_spacing REAL,
            document_purpose TEXT,
            confidence_scores TEXT,
            analysis_keywords TEXT,
            embedding TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()

def migrate_db():
    conn = sqlite3.connect('profiles.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT version FROM schema_version WHERE id = 1')
    result = cursor.fetchone()
    current_version = result[0] if result else 0
    
    if current_version < 1:
        current_version = 1
    
    if current_version < 2:
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN text_color TEXT DEFAULT "#000000"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN background_color TEXT DEFAULT "#FFFFFF"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN alignment TEXT DEFAULT "left"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN line_spacing REAL DEFAULT 1.33')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN document_purpose TEXT DEFAULT "general"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN confidence_scores TEXT DEFAULT "{}"')
        current_version = 2
    
    if current_version < 3:
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN name TEXT DEFAULT ""')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN fonts TEXT DEFAULT "{}"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN secondary_colors TEXT DEFAULT "[]"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN analysis_keywords TEXT DEFAULT "[]"')
        cursor.execute('ALTER TABLE style_profiles ADD COLUMN embedding TEXT DEFAULT "[]"')
        cursor.execute('UPDATE style_profiles SET fonts = ?', (json.dumps({'normal': 'Helvetica', 'bold': 'Helvetica-Bold', 'italic': 'Helvetica-Oblique'}),))
        current_version = 3
    
    cursor.execute('INSERT OR REPLACE INTO schema_version (id, version) VALUES (1, ?)', (CURRENT_SCHEMA_VERSION,))
    conn.commit()
    conn.close()

init_db()
migrate_db()

# Cache
response_cache = {}
session_messages = {}
html_previews = {}
analysis_cache = {}

def get_db_connection():
    conn = sqlite3.connect('profiles.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_default_style_profile():
    return {
        'name': 'Default',
        'fonts': {'normal': 'Helvetica', 'bold': 'Helvetica-Bold', 'italic': 'Helvetica-Oblique'},
        'font_size': 12,
        'tone': 'neutral',
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'structure': ['paragraphs'],
        'text_color': '#000000',
        'background_color': '#FFFFFF',
        'secondary_colors': [],
        'alignment': 'left',
        'line_spacing': 1.33,
        'document_purpose': 'general',
        'confidence_scores': {'font_size': 1.0, 'tone': 0.8, 'margins': 1.0, 'structure': 0.8, 'text_color': 1.0, 'background_color': 1.0, 'alignment': 0.9, 'line_spacing': 0.9, 'document_purpose': 0.7},
        'analysis_keywords': [],
        'embedding': []
    }

def parse_markdown_to_reportlab(text, style_profile=None):
    styles = getSampleStyleSheet()
    style_profile = style_profile or get_default_style_profile()
    fonts = style_profile['fonts']
    font_size = style_profile['font_size']
    alignment = {'left': 0, 'center': 1, 'right': 2, 'justified': 4}.get(style_profile['alignment'], 0)
    line_spacing = style_profile['line_spacing']
    text_color = colors.HexColor(style_profile['text_color'])
    background_color = colors.HexColor(style_profile['background_color'])
    
    available_fonts = pdfmetrics.getRegisteredFontNames()
    font_normal = fonts.get('normal', 'Helvetica')
    font_bold = fonts.get('bold', 'Helvetica-Bold')
    font_italic = fonts.get('italic', 'Helvetica-Oblique')
    
    if font_normal not in available_fonts:
        print(f"Fuente no válida '{font_normal}', usando 'Helvetica'")
        font_normal = 'Helvetica'
    if font_bold not in available_fonts:
        print(f"Fuente negrita no válida '{font_bold}', usando 'Helvetica-Bold'")
        font_bold = 'Helvetica-Bold'
    if font_italic not in available_fonts:
        print(f"Fuente cursiva no válida '{font_italic}', usando 'Helvetica-Oblique'")
        font_italic = 'Helvetica-Oblique'
    
    body_style = ParagraphStyle(
        name='Body',
        fontSize=font_size,
        leading=font_size * line_spacing,
        spaceAfter=8,
        fontName=font_normal,
        alignment=alignment,
        textColor=text_color,
        backColor=background_color
    )
    bold_style = ParagraphStyle(
        name='Bold',
        fontSize=font_size,
        leading=font_size * line_spacing,
        spaceAfter=8,
        fontName=font_bold,
        alignment=alignment,
        textColor=text_color,
        backColor=background_color
    )
    italic_style = ParagraphStyle(
        name='Italic',
        fontSize=font_size,
        leading=font_size * line_spacing,
        spaceAfter=8,
        fontName=font_italic,
        alignment=alignment,
        textColor=text_color,
        backColor=background_color
    )
    elements = []

    table_pattern = re.compile(r'\|(.+?)\|\n\|[-|:\s]+\|\n((?:\|.+?\|(?:\n|$))+)')
    text_parts = table_pattern.split(text)
    
    for i, part in enumerate(text_parts):
        if i % 3 == 2:
            rows = [row.strip().split('|')[1:-1] for row in part.strip().split('\n')]
            table_data = [[cell.strip() for cell in row] for row in rows]
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_normal),
                ('FONTSIZE', (0, 0), (-1, -1), font_size),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('LEADING', (0, 0), (-1, -1), font_size * line_spacing),
                ('TEXTCOLOR', (0, 0), (-1, -1), text_color),
                ('BACKCOLOR', (0, 0), (-1, -1), background_color),
            ]))
            elements.append(table)
            elements.append(Spacer(1, 0.2 * inch))
        elif i % 3 != 1:
            lines = part.split('\n')
            for line in lines:
                line = line.replace('\n', '<br />')
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
                if line.strip().startswith('- '):
                    line = f'• {line[2:]}'
                    elements.append(Paragraph(line, body_style))
                else:
                    style = bold_style if '**' in line else italic_style if '*' in line else body_style
                    elements.append(Paragraph(line, style))
                elements.append(Spacer(1, 0.1 * inch))
    
    return elements

def analyze_document(file, custom_profile_name=None):
    style_profile = {
        'name': custom_profile_name or f"Perfil-{datetime.now().strftime('%Y%m%d')}",
        'fonts': {'normal': 'Helvetica', 'bold': 'Helvetica-Bold', 'italic': 'Helvetica-Oblique'},
        'font_size': 12,
        'tone': 'neutral',
        'margins': {'top': 1, 'bottom': 1, 'left': 1, 'right': 1},
        'structure': [],
        'text_color': '#000000',
        'background_color': '#FFFFFF',
        'secondary_colors': [],
        'alignment': 'left',
        'line_spacing': 1.33,
        'document_purpose': 'general',
        'confidence_scores': {
            'font_size': 0.8,
            'tone': 0.7,
            'margins': 0.8,
            'structure': 0.7,
            'text_color': 0.8,
            'background_color': 0.8,
            'alignment': 0.7,
            'line_spacing': 0.7,
            'document_purpose': 0.6
        },
        'analysis_keywords': [],
        'embedding': []
    }
    
    content = ""
    summary = {
        'content_preview': '',
        'word_count': 0,
        'page_count': 1,
        'detected_structures': [],
        'visual_elements': {'images': 0, 'tables': 0},
        'tone_analysis': {'tone': 'neutral', 'confidence': 0.5},
        'purpose_analysis': {'purpose': 'general', 'confidence': 0.5}
    }
    
    try:
        file_hash = hashlib.md5(file.read()).hexdigest()
        file.seek(0)
        if file_hash in analysis_cache:
            style_profile, content, summary = analysis_cache[file_hash]
            return style_profile, content, summary
        
        if file.filename.endswith('.pdf'):
            with pdfplumber.open(file) as pdf:
                font_names = []
                font_styles = []
                font_sizes = []
                margins = {'top': [], 'bottom': [], 'left': [], 'right': []}
                text_colors = []
                background_colors = []
                alignments = []
                line_spacings = []
                structures = set()
                heading_levels = []
                image_count = 0
                
                max_pages = 5
                for page in pdf.pages[:min(len(pdf.pages), max_pages)]:
                    content += page.extract_text() or ""
                    
                    # Detectar imágenes
                    image_count += len(page.images)
                    
                    # Extraer fuentes incrustadas
                    for font in page.fonts:
                        font_name = font.get('BaseFont', 'Helvetica').split('-')[0]
                        font_name = re.sub(r'^[^a-zA-Z]+', '', font_name).lower()
                        if font.get('FontFile'):
                            try:
                                font_data = font['FontFile'].stream
                                pdfmetrics.registerFont(TTFont(font_name, BytesIO(font_data)))
                                print(f"Fuente personalizada registrada: {font_name}")
                            except Exception as e:
                                print(f"Error al registrar fuente: {e}")
                        
                        font_names.append(font_name)
                        font_styles.append(font.get('BaseFont', font_name))
                    
                    # Analizar caracteres
                    chars = page.chars
                    if chars:
                        raw_fonts = [char.get('fontname', 'Helvetica') for char in chars]
                        cleaned_fonts = [re.sub(r'^[^a-zA-Z]+', '', f).lower() for f in raw_fonts]
                        font_names.extend(cleaned_fonts)
                        font_sizes.extend([round(char.get('size', 12)) for char in chars])
                        text_colors.extend([char.get('non_stroking_color', (0, 0, 0)) for char in chars])
                    
                    # Calcular márgenes
                    if page.chars:
                        margins['top'].append(page.bbox[3] - max(c['y1'] for c in page.chars))
                        margins['bottom'].append(min(c['y0'] for c in page.chars) - page.bbox[1])
                        margins['left'].append(min(c['x0'] for c in page.chars) - page.bbox[0])
                        margins['right'].append(page.bbox[2] - max(c['x1'] for c in page.chars))
                    
                    # Detectar alineación y espaciado
                    if page.extract_text():
                        lines = page.extract_text().split('\n')
                        paragraph_spacings = []
                        for i in range(len(lines)-1):
                            try:
                                y0 = page.chars[i]['y0']
                                y1 = page.chars[i+1]['y0']
                                spacing = (y0 - y1) / page.chars[i].get('size', 12)
                                if spacing > 0 and not lines[i].strip().startswith(('-', '#', '|')):
                                    paragraph_spacings.append(spacing)
                            except IndexError:
                                continue
                        if paragraph_spacings:
                            line_spacings.append(statistics.mode(paragraph_spacings))
                        
                        x_positions = [c['x0'] for c in page.chars]
                        if x_positions:
                            x_variance = statistics.variance(x_positions) if len(x_positions) > 1 else 0
                            if x_variance < 10:
                                alignments.append('justified' if max(x_positions) - min(x_positions) > page.width * 0.8 else 'left')
                            elif max(x_positions) > page.width * 0.9:
                                alignments.append('right')
                            elif abs(max(x_positions) + min(x_positions) - page.width) < page.width * 0.1:
                                alignments.append('center')
                    
                    # Detectar estructuras
                    avg_size = statistics.mean(font_sizes) if font_sizes else 12
                    for char in page.chars:
                        size = char.get('size', 12)
                        if size > avg_size * 1.2:
                            heading_levels.append({'size': size, 'text': char['text']})
                    if page.extract_tables():
                        structures.add('tables')
                        summary['visual_elements']['tables'] += 1
                    if page.extract_text():
                        text = page.extract_text()
                        if re.search(r'^\s*[\•\-\*]\s+', text, re.MULTILINE):
                            structures.add('lists')
                        if re.search(r'^(#+)\s+', text, re.MULTILINE) or any(size > avg_size * 1.5 for size in font_sizes):
                            structures.add('headings')
                        if re.search(r'\n\n+', text):
                            structures.add('paragraphs')
                        if any(line.strip().startswith('>') for line in text.split('\n')):
                            structures.add('blockquotes')
                        if any(re.match(r'^\d+\.\s+', line) for line in text.split('\n')[-5:]):
                            structures.add('footnotes')
                
                # Normalizar fuentes
                if font_names:
                    font_counter = Counter(font_names)
                    raw_font = font_counter.most_common(1)[0][0]
                    print(f"Fuente detectada: {raw_font}")
                    font_mapping = {
                        'timesnewroman': 'Times-Roman',
                        'times new roman': 'Times-Roman',
                        'arial': 'Helvetica',
                        'arialm': 'Helvetica',
                        'helvetica': 'Helvetica',
                        'couriernew': 'Courier',
                        'courier new': 'Courier',
                        '': 'Helvetica'
                    }
                    style_profile['fonts']['normal'] = font_mapping.get(raw_font.lower(), 'Helvetica')
                    style_profile['confidence_scores']['font_size'] = font_counter.most_common(1)[0][1] / len(font_names)
                    
                    # Analizar variaciones
                    for style in font_styles:
                        if 'Bold' in style:
                            style_profile['fonts']['bold'] = style_profile['fonts']['normal'] + '-Bold'
                        if 'Italic' in style or 'Oblique' in style:
                            style_profile['fonts']['italic'] = style_profile['fonts']['normal'] + '-Oblique'
                
                if not font_names:
                    style_profile['fonts'] = {'normal': 'Helvetica', 'bold': 'Helvetica-Bold', 'italic': 'Helvetica-Oblique'}
                    style_profile['confidence_scores']['font_size'] = 0.5
                    print("No se detectaron fuentes, usando 'Helvetica'")
                
                if font_sizes:
                    size_counter = Counter(font_sizes)
                    style_profile['font_size'] = size_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['font_size'] = size_counter.most_common(1)[0][1] / len(font_sizes)
                
                if margins['top']:
                    margin_values = {key: Counter([round(v/72, 2) for v in margins[key]]) for key in margins}
                    for key in margins:
                        style_profile['margins'][key] = margin_values[key].most_common(1)[0][0]
                        style_profile['confidence_scores']['margins'] = min(1.0, len(margins[key]) / len(pdf.pages))
                
                if text_colors:
                    color_counter = Counter([tuple(c) if isinstance(c, (list, tuple)) else c for c in text_colors])
                    dominant_color = color_counter.most_common(1)[0][0]
                    if isinstance(dominant_color, (list, tuple)):
                        style_profile['text_color'] = '#{:02x}{:02x}{:02x}'.format(
                            int(dominant_color[0] * 255), int(dominant_color[1] * 255), int(dominant_color[2] * 255)
                        )
                    style_profile['secondary_colors'] = [
                        '#{:02x}{:02x}{:02x}'.format(int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
                        for c, _ in color_counter.most_common()[1:3]
                    ]
                    style_profile['confidence_scores']['text_color'] = color_counter.most_common(1)[0][1] / len(text_colors)
                
                if background_colors:
                    color_counter = Counter(background_colors)
                    dominant_bg = color_counter.most_common(1)[0][0]
                    style_profile['background_color'] = '#{:02x}{:02x}{:02x}'.format(*dominant_bg)
                
                if alignments:
                    alignment_counter = Counter(alignments)
                    style_profile['alignment'] = alignment_counter.most_common(1)[0][0]
                    style_profile['confidence_scores']['alignment'] = alignment_counter.most_common(1)[0][1] / len(alignments)
                
                if line_spacings:
                    style_profile['line_spacing'] = min(max(statistics.mode(line_spacings), 1.0), 2.0)
                    style_profile['confidence_scores']['line_spacing'] = len(line_spacings) / len(pdf.pages)
                
                style_profile['structure'] = list(structures)
                style_profile['structure'].append(f"heading_levels:{len(set(h['size'] for h in heading_levels))}")
                style_profile['confidence_scores']['structure'] = 0.9 if structures else 0.7
                
                summary['page_count'] = len(pdf.pages)
                summary['visual_elements']['images'] = image_count
        
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8', errors='ignore')
            file.seek(0)
            structures = set()
            md = markdown2.Markdown()
            parsed = md.convert(content)
            
            if '<ul>' in parsed or '<ol>' in parsed:
                structures.add('lists')
            if re.search(r'<h[1-6]', parsed):
                structures.add('headings')
            if re.search(r'<p>', parsed):
                structures.add('paragraphs')
            if re.search(r'<table>', parsed):
                structures.add('tables')
            if re.search(r'<blockquote>', parsed):
                structures.add('blockquotes')
            if '```' in content:
                style_profile['fonts']['normal'] = 'Courier'
                style_profile['fonts']['bold'] = 'Courier-Bold'
                style_profile['fonts']['italic'] = 'Courier-Oblique'
                style_profile['confidence_scores']['font_size'] = 0.7
            
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.8 if structures else 0.6
        
        elif file.filename.endswith('.docx'):
            doc = Document(file)
            content = '\n'.join(p.text for p in doc.paragraphs)
            structures = set()
            for p in doc.paragraphs:
                if p.style.name.startswith('Heading'):
                    structures.add('headings')
                if p.style.name == 'List Bullet' or p.style.name == 'List Number':
                    structures.add('lists')
                if p.text.strip():
                    structures.add('paragraphs')
            for table in doc.tables:
                structures.add('tables')
                summary['visual_elements']['tables'] += 1
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.8 if structures else 0.6
        
        elif file.filename.endswith(('.png', '.jpg', '.jpeg')):
            content = pytesseract.image_to_string(Image.open(file))
            structures = set(['paragraphs'])
            style_profile['structure'] = list(structures)
            style_profile['confidence_scores']['structure'] = 0.6
            summary['visual_elements']['images'] = 1
        
        else:
            raise ValueError("Formato no soportado")
        
        # Analizar tono y propósito
        if content.strip():
            try:
                response = client.chat.completions.create(
                    model="gpt-4o" if client.models.list().data else "gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """
                            Analiza el texto y devuelve un JSON con:
                            - 'tone': tono (e.g., formal, informal, técnico, académico).
                            - 'document_purpose': propósito (e.g., informe, carta, manual, acta, propuesta).
                            - 'confidence': confianza (0.0 a 1.0).
                            - 'keywords': palabras clave del análisis.
                            """
                        },
                        {"role": "user", "content": content[:4000]}
                    ],
                    max_tokens=500
                )
                analysis = json.loads(response.choices[0].message.content)
                style_profile['tone'] = analysis.get('tone', 'neutral')
                style_profile['document_purpose'] = analysis.get('document_purpose', 'general')
                style_profile['confidence_scores']['tone'] = analysis.get('confidence', 0.7)
                style_profile['confidence_scores']['document_purpose'] = analysis.get('confidence', 0.7)
                style_profile['analysis_keywords'] = analysis.get('keywords', [])
                
                summary['tone_analysis'] = {
                    'tone': style_profile['tone'],
                    'confidence': style_profile['confidence_scores']['tone']
                }
                summary['purpose_analysis'] = {
                    'purpose': style_profile['document_purpose'],
                    'confidence': style_profile['confidence_scores']['document_purpose']
                }
            
            except Exception as e:
                print(f"Error en análisis de tono: {e}")
                formal_keywords = ['estimado', 'cordialmente', 'respecto']
                technical_keywords = ['implementación', 'arquitectura', 'sistema']
                purpose_keywords = {
                    'acta': ['acta', 'reunión', 'acuerdos'],
                    'propuesta': ['propuesta', 'objetivos', 'presupuesto'],
                    'manual': ['manual', 'instrucciones', 'guía']
                }
                if any(kw in content.lower() for kw in formal_keywords):
                    style_profile['tone'] = 'formal'
                    style_profile['confidence_scores']['tone'] = 0.7
                elif any(kw in content.lower() for kw in technical_keywords):
                    style_profile['tone'] = 'technical'
                    style_profile['confidence_scores']['tone'] = 0.7
                for purpose, kws in purpose_keywords.items():
                    if any(kw in content.lower() for kw in kws):
                        style_profile['document_purpose'] = purpose
                        style_profile['confidence_scores']['document_purpose'] = 0.7
                        break
                
                summary['tone_analysis'] = {
                    'tone': style_profile['tone'],
                    'confidence': style_profile['confidence_scores']['tone']
                }
                summary['purpose_analysis'] = {
                    'purpose': style_profile['document_purpose'],
                    'confidence': style_profile['confidence_scores']['document_purpose']
                }
            
            # Generar embeddings
            try:
                embedding = client.embeddings.create(input=content[:4000], model="text-embedding-ada-002").data[0].embedding
                style_profile['embedding'] = embedding
            except Exception as e:
                print(f"Error al generar embeddings: {e}")
        else:
            style_profile['tone'] = 'neutral'
            style_profile['document_purpose'] = 'general'
            style_profile['confidence_scores']['tone'] = 0.5
            style_profile['confidence_scores']['document_purpose'] = 0.5
        
        # Completar resumen
        summary['content_preview'] = content[:500]
        summary['word_count'] = len(content.split())
        summary['detected_structures'] = style_profile['structure']
        
        analysis_cache[file_hash] = (style_profile, content, summary)
    
    except Exception as e:
        print(f"Error al analizar documento: {e}")
        style_profile = get_default_style_profile()
        content = ""
        summary = {
            'content_preview': '',
            'word_count': 0,
            'page_count': 1,
            'detected_structures': [],
            'visual_elements': {'images': 0, 'tables': 0},
            'tone_analysis': {'tone': 'neutral', 'confidence': 0.5},
            'purpose_analysis': {'purpose': 'general', 'confidence': 0.5}
        }
    
    return style_profile, content, summary

@app.route('/')
@login_required
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, email, password FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            login_user(User(user['id'], user['email']))
            return redirect(url_for('index'))
        else:
            flash('Correo o contraseña incorrectos', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (email, password) VALUES (?, ?)', (email, hashed_password))
            conn.commit()
            conn.close()
            flash('Registro exitoso. Por favor, inicia sesión.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('El correo ya está registrado', 'error')
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/generate', methods=['POST'])
@login_required
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
        message_history = data.get('message_history', [])

        if not prompt:
            return jsonify({'error': 'El prompt está vacío'}), 400

        cache_key = f"{current_user.id}:{session_id}:{prompt}:{doc_type}:{tone}:{length}:{language}:{style_profile_id}"
        if cache_key in response_cache:
            return jsonify({
                'document': response_cache[cache_key],
                'content_type': 'html' if doc_type == 'html' or 'html' in prompt.lower() else 'pdf'
            })

        if session_id not in session_messages:
            session_messages[session_id] = []

        style_profile = get_style_profile(style_profile_id)
        print("style_profile en /generate:", style_profile)

        effective_tone = style_profile['tone'] if style_profile.get('tone') else tone
        structure = style_profile['structure'] if style_profile.get('structure') else ['paragraphs']
        structure_instruction = f"Usa una estructura que incluya: {', '.join(s for s in structure if not s.startswith('heading_levels'))}."

        is_explanatory = re.search(r'^(¿Quién es|¿Qué es|explicar|detallar)\b', prompt, re.IGNORECASE) is not None
        is_html = doc_type == 'html' or re.search(r'\b(html|página web|sitio web)\b', prompt, re.IGNORECASE) is not None
        
        if is_explanatory:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown con una estructura clara para explicaciones.
            - Tipo de documento: {doc_type}.
            - Tono: {effective_tone}.
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language}.
            - {structure_instruction}
            - Usa la siguiente estructura en Markdown:
              ```markdown
              # [Tema o Nombre]
              
              ## Introducción
              [Párrafo breve presentando el tema o persona.]
              
              ## Detalles Principales
              - [Clave 1]: [Valor o descripción]
              - [Clave 2]: [Valor o descripción]
              - ...
              
              ## Contexto Adicional
              [Párrafos detallando información relevante.]
              
              ## Conclusión
              [Resumen de la relevancia o importancia.]
              ```
            - Usa encabezados, listas, negritas, y tablas cuando sea apropiado.
            - Considera el contexto de los mensajes anteriores.
            """
        elif is_html:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera páginas web en formato HTML con CSS y JavaScript si es necesario.
            - Tipo de documento: página web (HTML).
            - Tono: {effective_tone}.
            - Longitud: {length} (corto: ~100 líneas, medio: ~300 líneas, largo: ~600 líneas).
            - Idioma: {language}.
            - Genera código HTML completo con:
              - Estructura semántica (<header>, <main>, <footer>).
              - Estilos CSS internos adaptados al tono.
              - JavaScript opcional si el prompt lo requiere.
              - Usa un diseño responsivo.
            - Considera el contexto de los mensajes anteriores.
            """
        else:
            system_prompt = f"""
            Eres GarBotGPT, un asistente que genera documentos profesionales en formato Markdown.
            - Tipo de documento: {doc_type}.
            - Tono: {effective_tone}.
            - Longitud: {length} (corto: ~100 palabras, medio: ~300 palabras, largo: ~600 palabras).
            - Idioma: {language}.
            - {structure_instruction}
            - Usa encabezados, listas, negritas, y tablas cuando sea apropiado.
            - Considera el contexto de los mensajes anteriores.
            """

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(message_history[-10:])
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )

        document = response.choices[0].message.content
        response_cache[cache_key] = document

        session_messages[session_id].append({"role": "user", "content": prompt})
        session_messages[session_id].append({"role": "assistant", "content": document})

        return jsonify({
            'document': document,
            'content_type': 'html' if is_html else 'pdf'
        })
    except openai.AuthenticationError:
        return jsonify({'error': 'Error de autenticación con OpenAI.'}), 401
    except openai.RateLimitError:
        return jsonify({'error': 'Límite de solicitudes alcanzado.'}), 429
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_style_profile(profile_id):
    if not profile_id:
        return get_default_style_profile()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM style_profiles WHERE id = ? AND user_id = ?', (profile_id, current_user.id))
    profile = cursor.fetchone()
    conn.close()
    if profile:
        return {
            'name': profile['name'],
            'fonts': json.loads(profile['fonts']),
            'font_size': profile['font_size'],
            'tone': profile['tone'],
            'margins': json.loads(profile['margins']),
            'structure': json.loads(profile['structure']),
            'text_color': profile['text_color'],
            'background_color': profile['background_color'],
            'secondary_colors': json.loads(profile['secondary_colors']),
            'alignment': profile['alignment'],
            'line_spacing': profile['line_spacing'],
            'document_purpose': profile['document_purpose'],
            'confidence_scores': json.loads(profile['confidence_scores']),
            'analysis_keywords': json.loads(profile['analysis_keywords']),
            'embedding': json.loads(profile['embedding'])
        }
    return get_default_style_profile()

@app.route('/generate_preview', methods=['POST'])
@login_required
def generate_preview():
    try:
        data = request.json
        content = data.get('content', '')
        content_type = data.get('content_type', 'pdf')
        style_profile_id = data.get('style_profile_id', None)
        if not content:
            return jsonify({'error': 'El contenido está vacío'}), 400

        if content_type == 'html':
            preview_id = f"{current_user.id}:{datetime.now().timestamp()}"
            html_previews[preview_id] = content
            return jsonify({'preview_id': preview_id, 'content_type': 'html'})
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer, 
            pagesize=letter, 
            rightMargin=0.75*inch, 
            leftMargin=0.75*inch, 
            topMargin=inch, 
            bottomMargin=0.75*inch
        )
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(name='Title', fontSize=18, leading=22, spaceAfter=12, fontName='Helvetica-Bold')
        subtitle_style = ParagraphStyle(name='Subtitle', fontSize=10, leading=14, spaceAfter=10, fontName='Helvetica-Oblique')

        story = []
        story.append(Paragraph("Documento Generado por GarBotGPT", title_style))
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y %H:%M')}", subtitle_style))
        story.append(Spacer(1, 0.3 * inch))
        
        style_profile = get_style_profile(style_profile_id)
        print("Aplicando style_profile en generate_preview:", style_profile)
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

@app.route('/preview_html/<preview_id>', methods=['GET'])
@login_required
def preview_html(preview_id):
    if preview_id in html_previews and preview_id.startswith(f"{current_user.id}:"):
        return Response(html_previews[preview_id], mimetype='text/html')
    return jsonify({'error': 'Vista previa no encontrada'}), 404

@app.route('/analyze_document', methods=['POST'])
@login_required
def analyze_document_endpoint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se proporcionó ningún archivo'}), 400
        
        file = request.files['file']
        custom_profile_name = request.form.get('profile_name', None)
        if not (file.filename.endswith(('.pdf', '.txt', '.docx', '.png', '.jpg', '.jpeg'))):
            return jsonify({'error': 'Formato no soportado. Usa PDF, TXT, DOCX o imágenes.'}), 400
        
        style_profile, content, summary = analyze_document(file, custom_profile_name)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO style_profiles (
                user_id, name, fonts, font_size, tone, margins, structure, 
                text_color, background_color, secondary_colors, alignment, 
                line_spacing, document_purpose, confidence_scores, analysis_keywords, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            current_user.id,
            style_profile['name'],
            json.dumps(style_profile['fonts']),
            style_profile['font_size'],
            style_profile['tone'],
            json.dumps(style_profile['margins']),
            json.dumps(style_profile['structure']),
            style_profile['text_color'],
            style_profile['background_color'],
            json.dumps(style_profile['secondary_colors']),
            style_profile['alignment'],
            style_profile['line_spacing'],
            style_profile['document_purpose'],
            json.dumps(style_profile['confidence_scores']),
            json.dumps(style_profile['analysis_keywords']),
            json.dumps(style_profile['embedding'])
        ))
        profile_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return jsonify({
            'style_profile_id': str(profile_id),
            'style_profile': style_profile,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/style_profiles', methods=['GET'])
@login_required
def get_style_profiles():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM style_profiles WHERE user_id = ?', (current_user.id,))
    profiles = cursor.fetchall()
    conn.close()
    
    result = {}
    for profile in profiles:
        result[str(profile['id'])] = {
            'name': profile['name'],
            'fonts': json.loads(profile['fonts']),
            'font_size': profile['font_size'],
            'tone': profile['tone'],
            'margins': json.loads(profile['margins']),
            'structure': json.loads(profile['structure']),
            'text_color': profile['text_color'],
            'background_color': profile['background_color'],
            'secondary_colors': json.loads(profile['secondary_colors']),
            'alignment': profile['alignment'],
            'line_spacing': profile['line_spacing'],
            'document_purpose': profile['document_purpose'],
            'confidence_scores': json.loads(profile['confidence_scores']),
            'analysis_keywords': json.loads(profile['analysis_keywords']),
            'embedding': json.loads(profile['embedding'])
        }
    
    return jsonify(result)

@app.route('/style_profiles/<profile_id>', methods=['DELETE'])
@login_required
def delete_style_profile(profile_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM style_profiles WHERE id = ? AND user_id = ?', (profile_id, current_user.id))
    conn.commit()
    conn.close()
    
    if cursor.rowcount > 0:
        return jsonify({'message': 'Perfil de estilo eliminado'})
    return jsonify({'error': 'Perfil no encontrado'}), 404

@app.route('/purge_style_profiles', methods=['DELETE'])
@login_required
def purge_style_profiles():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM style_profiles WHERE user_id = ?', (current_user.id,))
        conn.commit()
        conn.close()
        return jsonify({'message': 'Todos los perfiles de estilo han sido eliminados'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))