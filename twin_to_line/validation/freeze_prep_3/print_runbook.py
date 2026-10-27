"""Render the exact section-7 text as a single Letter page (development tooling only)."""
from html import escape
from pathlib import Path
import re
import sys

from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parents[2]
font = sys.argv[1]
pdfmetrics.registerFont(TTFont('Body', font))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Body', italic='Body', boldItalic='Body')
text = (HERE / 'validation/freeze_prep_3/section7.md').read_text()
path = HERE / 'monday/MONDAY_ONE_PAGE.pdf'
style = ParagraphStyle('body', fontName='Body', fontSize=10.2, leading=12.4, spaceAfter=7, alignment=TA_LEFT)
title = ParagraphStyle('title', parent=style, fontSize=14, leading=17, spaceAfter=10)
story = []
for block in text.strip().split('\n\n'):
    is_title = block.startswith('## ')
    block = ' '.join(line.strip() for line in block.removeprefix('## ').splitlines())
    block = block.replace('–', '-').replace('—', '-')
    block = escape(block)
    block = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', block)
    block = re.sub(r'`(.*?)`', r'<font name="Courier" size="9">\1</font>', block)
    story.append(Paragraph(block, title if is_title else style))
doc = SimpleDocTemplate(str(path), pagesize=(612,792), leftMargin=32,rightMargin=32,topMargin=30,bottomMargin=30,
                        title='Monday operator runbook - section 7', author='', creator='Twin-to-Line')
doc.build(story)
pdf = pdfium.PdfDocument(str(path))
assert len(pdf) == 1, f'runbook exceeds one page: {len(pdf)}'
preview = HERE / 'results/freeze_prep_3/round1/one_page.png'
pdf[0].render(scale=1.8).to_pil().save(preview)
print('MONDAY_ONE_PAGE.pdf: 1 Letter page; body 10.2 pt, leading 12.4 pt')
