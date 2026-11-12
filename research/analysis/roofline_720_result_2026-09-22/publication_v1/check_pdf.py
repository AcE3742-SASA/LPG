"""Check vector PDF dimensions/fonts and make an A4 proof at actual size."""
from pathlib import Path
import hashlib
import io
import json
from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, FloatObject
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import reportlab

ROOT = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('ProofSans', str(Path(reportlab.__file__).parent/'fonts/Vera.ttf')))
layouts = json.loads((ROOT/'layout_checks.json').read_text())
proof = PdfWriter()
checks = []
for layout in layouts:
    name = layout['figure']
    reader = PdfReader(ROOT/f'{name}.pdf')
    assert len(reader.pages) == 1
    page = reader.pages[0]
    dimensions = [float(page.mediabox.width)/mm, float(page.mediabox.height)/mm]
    assert max(abs(a-b) for a,b in zip(dimensions, [layout['width_mm'], layout['height_mm']])) < .001
    resources = page['/Resources']
    assert not any(obj.get_object().get('/Subtype') == '/Image' for obj in resources.get('/XObject', {}).values())
    fonts = []
    for value in resources['/Font'].values():
        font = value.get_object()
        descendants = font.get('/DescendantFonts', [font])
        for child in descendants:
            descriptor = child.get_object()['/FontDescriptor']
            assert '/FontFile2' in descriptor or '/FontFile3' in descriptor
        fonts.append(str(font['/BaseFont']))
    sizes = set()
    text = page.extract_text(visitor_text=lambda text, cm, tm, font, size: sizes.add(size) if text.strip() else None)
    assert sizes <= {7.0, 10.0} and 10.0 in sizes, (name, sizes)
    for value in ['6.14', '35.02', '262.3']:
        assert value in text, (name, value)
    # The proof copy changes RGB paint to gray; submitted vector PDFs remain untouched.
    source_digest = hashlib.sha256((ROOT/f'{name}.pdf').read_bytes()).hexdigest()
    gray = PdfWriter()
    gray.add_page(page)
    gray_page = gray.pages[0]
    stream = ContentStream(gray_page.get_contents(), gray)
    for index, (operands, operator) in enumerate(stream.operations):
        if operator in (b'rg', b'RG'):
            luminance = sum(float(v)*weight for v, weight in zip(operands, (.299, .587, .114)))
            stream.operations[index] = ([FloatObject(luminance)], b'g' if operator == b'rg' else b'G')
    gray_page.replace_contents(stream)
    header = io.BytesIO()
    drawing = canvas.Canvas(header, pagesize=A4)
    drawing.setFont('ProofSans', 10)
    drawing.drawString(14*mm, A4[1]-14*mm, 'ACTUAL-SIZE GRAYSCALE PROOF | print at 100%; disable fit-to-page')
    drawing.drawString(14*mm, A4[1]-21*mm,
                       f'{name} | {layout["width_mm"]} x {layout["height_mm"]} mm | base labels: 10 pt')
    drawing.setLineWidth(.5)
    drawing.line(14*mm, 12*mm, 64*mm, 12*mm)
    drawing.drawString(14*mm, 7*mm, '50 mm scale check')
    drawing.save()
    proof_page = PdfReader(header).pages[0]
    x = (A4[0]-float(gray_page.mediabox.width))/2
    y = A4[1]-31*mm-float(gray_page.mediabox.height)
    assert y > 20*mm
    proof_page.merge_transformed_page(gray_page, Transformation().translate(x, y))
    proof.add_page(proof_page)
    assert hashlib.sha256((ROOT/f'{name}.pdf').read_bytes()).hexdigest() == source_digest
    checks.append(dict(file=f'{name}.pdf', dimensions_mm=dimensions, all_fonts_embedded=True,
                       fonts=fonts, embedded_raster_images=0, font_sizes_pt=sorted(sizes),
                       values_checked=True, sha256=source_digest))
with (ROOT/'proof_actual_size.pdf').open('wb') as output:
    proof.write(output)
result = dict(status='PASS', pdf_checks=checks, proof_pages=len(proof.pages),
              proof_paper='A4', proof_scale=1.0,
              typography='Base labels 10 pt; logarithmic superscripts 7 pt')
(ROOT/'publication_validation.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
