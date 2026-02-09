import os
import platform
import re
from pathlib import Path
from typing import Optional
import logging

from openai import OpenAI
from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

logger = logging.getLogger("document")
if os.getenv("DEBUG", "false").lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.DEBUG)
else:
    logger.addHandler(logging.NullHandler())

try:
    from document_reader import DocumentReader
    READER_AVAILABLE = True
except ImportError:
    READER_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.colors import HexColor
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

import json as _json

# --- Company Profile ---
# Priority: company.json file > environment variables > defaults below

_COMPANY_DEFAULTS = {
    "name": "ООО «АйТи-Солюшнс Про»",
    "full_name": "Общество с ограниченной ответственностью «АйТи-Солюшнс Про»",
    "details": "ИНН 1234567890, КПП 123456789, ОГРН 112233445566",
    "address": "г. Москва, ул. Примерная, д. 10, оф. 505",
    "contacts": "Тел: +7 (999) 000-00-00, Email: info@it-solutions.pro",
    "director": "Иванов Иван Иванович",
    "experience": "7 лет на рынке разработки ПО, автоматизации и системной интеграции.",
    "stack": "Python, React, Machine Learning, Big Data, Highload-системы.",
    "advantages": "Собственный штат разработчиков, гарантия 12 месяцев, техподдержка 24/7.",
}

_ENV_MAP = {
    "name":       "COMPANY_NAME",
    "full_name":  "COMPANY_FULL_NAME",
    "details":    "COMPANY_DETAILS",
    "address":    "COMPANY_ADDRESS",
    "contacts":   "COMPANY_CONTACTS",
    "director":   "COMPANY_DIRECTOR",
    "experience": "COMPANY_EXPERIENCE",
    "stack":      "COMPANY_STACK",
    "advantages": "COMPANY_ADVANTAGES",
}

COMPANY_JSON_PATH = Path("company.json")


def load_company_profile() -> dict:
    """Load company profile: company.json > env vars > defaults."""
    profile = dict(_COMPANY_DEFAULTS)

    # Layer 1: override from environment variables
    for field, env_key in _ENV_MAP.items():
        val = os.getenv(env_key)
        if val:
            profile[field] = val

    # Layer 2: override from company.json (highest priority)
    if COMPANY_JSON_PATH.exists():
        try:
            with open(COMPANY_JSON_PATH, 'r', encoding='utf-8') as f:
                saved = _json.load(f)
            for field in _COMPANY_DEFAULTS:
                if field in saved and saved[field]:
                    profile[field] = saved[field]
        except (OSError, _json.JSONDecodeError) as e:
            logger.warning(f"Failed to read {COMPANY_JSON_PATH}: {e}")

    return profile


def save_company_profile(profile: dict) -> bool:
    """Save company profile to company.json. Returns True on success."""
    # Validate: only allow known keys
    clean = {k: str(v).strip() for k, v in profile.items() if k in _COMPANY_DEFAULTS and v}
    try:
        with open(COMPANY_JSON_PATH, 'w', encoding='utf-8') as f:
            _json.dump(clean, f, ensure_ascii=False, indent=2)
        return True
    except OSError as e:
        logger.error(f"Failed to save {COMPANY_JSON_PATH}: {e}")
        return False


# Module-level profile (loaded once on import, refreshed by DocumentGenerator)
COMPANY_PROFILE = load_company_profile()

class DocumentGenerator:
    def __init__(self, api_key: str = None, base_url: str = "https://api.proxyapi.ru/openai/v1"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"), base_url=base_url)
        self.model = "gpt-4o-mini"
        self.reader = DocumentReader() if READER_AVAILABLE else None
        self.font_name = self._register_font()

    @property
    def company(self) -> dict:
        """Always return fresh company profile (picks up saves via API)."""
        return load_company_profile()
        self.font_name = self._register_font()

    def _register_font(self):
        if not PDF_AVAILABLE:
            return None

        system = platform.system()
        paths = []

        if system == 'Windows':
            paths = ['C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/tahoma.ttf', 'C:/Windows/Fonts/calibri.ttf']
        elif system == 'Darwin':
            paths = ['/Library/Fonts/Arial.ttf']
        else:
            paths = [
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/TTF/Arial.ttf',
                '/usr/share/fonts/liberation/LiberationSans-Regular.ttf'
            ]

        for path in paths:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont('CustomFont', path))
                    return 'CustomFont'
                except Exception:
                    continue
        return 'Helvetica'

    def _read_files(self, directory: str) -> str:
        if not self.reader or not directory or not os.path.exists(directory):
            return ""
        try:
            print(f"   (Анализ файлов из папки {os.path.basename(directory)}...)")
            return self.reader.read_directory(directory)
        except Exception as e:
            logger.debug(f"File read error: {e}")
            return ""

    def _call_ai(self, prompt: str, max_tokens: int = 4000) -> str:
        cp = self.company
        system_msg = (
            f"Ты — опытный тендерный специалист компании {cp['name']}. "
            f"Твоя задача — готовить профессиональную документацию для участия в закупках. "
            f"Используй официально-деловой стиль. "
            f"Информация о компании: {cp['experience']} {cp['stack']}. "
            f"Преимущества: {cp['advantages']}. "
            f"Подписант: Генеральный директор {cp['director']}."
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.4
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.debug(f"AI Error: {e}")
            return "Ошибка генерации текста. Проверьте баланс API или настройки."

    def generate_proposal(self, data: dict, files_dir: str = "") -> str:
        cp = self.company
        files_content = self._read_files(files_dir)
        prompt = (
            f"Подготовь КОММЕРЧЕСКОЕ ПРЕДЛОЖЕНИЕ (КП) для тендера.\n\n"
            f"ЗАКАЗЧИК: {data['customer']}\n"
            f"ТЕНДЕР: {data['title']} (№ {data['number']})\n"
            f"БЮДЖЕТ: {data['price']}\n"
            f"КРАТКОЕ ОПИСАНИЕ: {data['description'][:1500]}\n"
            f"КОНТЕКСТ ИЗ ФАЙЛОВ (ТЗ): {files_content[:10000]}\n\n"
            f"Требования к документу:\n"
            f"1. Структура:\n"
            f"   - Шапка (от кого: {cp['full_name']}, {cp['details']}).\n"
            f"   - Заголовок (Коммерческое предложение на ...).\n"
            f"   - Уважаемые коллеги (обращение к заказчику).\n"
            f"   - Понимание задачи (кратко опиши, что нужно сделать, исходя из ТЗ).\n"
            f"   - Наше решение (предложи техническое решение на стеке {cp['stack']}, этапы работ).\n"
            f"   - Стоимость и сроки (если точных нет, напиши 'Согласно ТЗ' или предложи расчет).\n"
            f"   - Почему мы (используй наши преимущества: {cp['advantages']}).\n"
            f"   - Заключение и подпись.\n"
            f"2. Стиль: Строгий, убедительный, без воды.\n"
            f"3. Форматирование: Используй Markdown (заголовки #, жирный шрифт **)."
        )
        return self._call_ai(prompt, 4000)

    def generate_letter(self, data: dict, files_dir: str = "") -> str:
        cp = self.company
        prompt = (
            f"Подготовь СОПРОВОДИТЕЛЬНОЕ ПИСЬМО к заявке на участие в тендере.\n"
            f"Тендер: {data['title']} (№ {data['number']})\n"
            f"Заказчик: {data['customer']}\n\n"
            f"Структура:\n"
            f"1. Шапка (на бланке организации).\n"
            f"2. Исх. номер и дата (текущая).\n"
            f"3. Текст: Мы, {cp['full_name']}, изучили документацию, согласны со всеми условиями и предлагаем свои услуги.\n"
            f"4. Гарантируем качество и соблюдение сроков.\n"
            f"5. Приложения (перечислить: Коммерческое предложение, Опись документов и др.).\n"
            f"6. Подпись ({cp['director']})."
        )
        return self._call_ai(prompt, 1500)

    def generate_requirements(self, data: dict, files_dir: str = "") -> str:
        cp = self.company
        files_content = self._read_files(files_dir)
        reqs = data.get('requirements') or data.get('description', '')
        prompt = (
            f"Составь ТАБЛИЦУ СООТВЕТСТВИЯ ТРЕБОВАНИЯМ (Форма 2 / Техническое предложение).\n"
            f"Тендер: {data['title']}\n"
            f"Выдержка из требований заказчика: {reqs[:3000]}\n"
            f"Контекст из файлов документации: {files_content[:10000]}\n\n"
            f"Задача:\n"
            f"1. Проанализируй требования и напиши ответ на каждый пункт.\n"
            f"2. Формат: Таблица (или список), где есть 'Требование заказчика' и 'Предложение участника ({cp['name']})'.\n"
            f"3. В графе 'Предложение' пиши конкретные характеристики, подтверждающие соответствие (слово 'Соответствует' используй, но добавляй детали).\n"
            f"4. Если в требованиях указаны ГОСТы или конкретные параметры, подтверждай их."
        )
        return self._call_ai(prompt, 4000)

    def save_docx(self, content: str, filename: str, folder: str = "outputs") -> str:
        Path(folder).mkdir(exist_ok=True)
        path = os.path.join(folder, f"{filename}.docx")

        doc = Document()
        style = doc.styles['Normal']
        style.font.name = 'Arial'
        style.font.size = Pt(11)

        for line in content.split('\n'):
            line = line.strip()
            if not line: continue

            if line.startswith('# '):
                doc.add_heading(line.lstrip('# ').strip(), level=1)
            elif line.startswith('## '):
                doc.add_heading(line.lstrip('# ').strip(), level=2)
            elif line.startswith('### '):
                doc.add_heading(line.lstrip('# ').strip(), level=3)
            elif line.startswith(('-', '*', '•')):
                p = doc.add_paragraph(style='List Bullet')
                self._format_text(p, line.lstrip('-*• '))
            else:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                self._format_text(p, line)
        
        doc.save(path)
        return path

    def _format_text(self, paragraph, text):
        parts = re.split(r'(\*\*.*\*\*)', text)
        for part in parts:
            if part.startswith('**') and part.endswith('**'):
                run = paragraph.add_run(part[2:-2])
                run.font.bold = True
            else:
                paragraph.add_run(part)

    def save_pdf(self, content: str, filename: str, folder: str = "outputs") -> Optional[str]:
        if not PDF_AVAILABLE: return None
        Path(folder).mkdir(exist_ok=True)
        path = os.path.join(folder, f"{filename}.pdf")

        doc = SimpleDocTemplate(
            path, 
            pagesize=A4,
            rightMargin=20*mm, leftMargin=20*mm, 
            topMargin=20*mm, bottomMargin=20*mm
        )
        
        styles = getSampleStyleSheet()
        base_style = ParagraphStyle(
            'Base', 
            parent=styles['Normal'], 
            fontName=self.font_name, 
            fontSize=10, 
            leading=14, 
            alignment=TA_JUSTIFY, 
            spaceAfter=6
        )
        
        h1_style = ParagraphStyle('H1', parent=base_style, fontSize=14, leading=18, spaceBefore=12, spaceAfter=6, textColor=HexColor('#003366'), fontName=self.font_name)
        h2_style = ParagraphStyle('H2', parent=base_style, fontSize=12, leading=16, spaceBefore=10, spaceAfter=6, fontName=self.font_name,  textColor=HexColor('#003366'))
        
        story = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line: 
                story.append(Spacer(1, 4))
                continue

            clean_line = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            fmt_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)

            if line.startswith('# '):
                story.append(Paragraph(fmt_line.lstrip('# '), h1_style))
            elif line.startswith('## '):
                story.append(Paragraph(fmt_line.lstrip('# '), h2_style))
            elif line.startswith('### '):
                story.append(Paragraph(fmt_line.lstrip('# '), h2_style))
            elif line.startswith(('-', '*', '•')):
                story.append(Paragraph(f"• {fmt_line.lstrip('-*• ')}", base_style))
            else:
                story.append(Paragraph(fmt_line, base_style))

        try:
            doc.build(story)
            return path
        except Exception as e:
            logger.debug(f"PDF Save error: {e}")
            return None