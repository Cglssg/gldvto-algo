import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF

# 指定存放 SVG 文件的文件夹路径（请根据你的实际情况修改）
svg_folder = r""

# 获取文件夹中所有的 .svg 文件
svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith(".svg")]

# 遍历并转换每一个 SVG 文件
for svg_file in svg_files:
    # 拼接完整的文件路径
    svg_path = os.path.join(svg_folder, svg_file)

    # 生成对应的 PDF 文件名（将 .svg 后缀替换为 .pdf）
    pdf_name = svg_file.replace(".svg", ".pdf")
    pdf_path = os.path.join(svg_folder, pdf_name)

    # 1. 使用 svg2rlg 读取并解析 SVG 文件
    drawing = svg2rlg(svg_path)

    # 2. 使用 renderPDF 将解析后的绘图对象保存为 PDF
    renderPDF.drawToFile(drawing, pdf_path)

    print(f"已成功转换: {svg_file} -> {pdf_name}")

print("所有文件转换完成！")