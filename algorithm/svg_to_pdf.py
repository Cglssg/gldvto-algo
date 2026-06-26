import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

svg_folder = r""

svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith(".svg")]

for svg_file in svg_files:
    svg_path = os.path.join(svg_folder, svg_file)
    pdf_name = svg_file.replace(".svg", ".pdf")
    pdf_path = os.path.join(svg_folder, pdf_name)
    drawing = svg2rlg(svg_path)
    renderPDF.drawToFile(drawing, pdf_path)

    print(f"已成功转换: {svg_file} -> {pdf_name}")

print("所有文件转换完成！")