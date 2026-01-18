
import fitz  # PyMuPDF

def _get_table_bboxes(page):
    """
    Extract table bounding boxes from a PDF page using PyMuPDF.
    Args:
        page: PyMuPDF page object.
    Returns:
        List of fitz.Rect objects representing table bounding boxes.
    """
    tables = page.find_tables()
    return [fitz.Rect(t.bbox) for t in tables]

def _get_image_bboxes(page):
    """
    Extract image bounding boxes from a PDF page using PyMuPDF.
    Args:
        page: PyMuPDF page object.
    Returns:
        List of fitz.Rect objects representing image bounding boxes.
    """
    image_bboxes = []
    for img in page.get_images():
        rects = page.get_image_rects(img[0])
        image_bboxes.extend(rects)
    return image_bboxes

def _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
    """
    Determine if a text block should be filtered out (header, footer, table, or image).
    Args:
        bbox: fitz.Rect of the block.
        header_height: Height of the header region.
        footer_y: Y-coordinate of the footer region start.
        table_bboxes: List of table bounding boxes.
        image_bboxes: List of image bounding boxes.
    Returns:
        True if the block should be filtered, False otherwise.
    """
    if bbox.y1 < header_height or bbox.y0 > footer_y:
        return True
    block_center = fitz.Point((bbox.x0 + bbox.x1)/2, (bbox.y0 + bbox.y1)/2)
    if any(block_center in t_bbox for t_bbox in table_bboxes):
        return True
    if any(block_center in i_bbox for i_bbox in image_bboxes):
        return True
    return False

def _process_line(line, page_num, page_height, page_width):
    """
    Process a line from a text block and extract text and character coordinates.
    Args:
        line: Line dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        Tuple of (line_text, line_chars) where line_text is the string and line_chars is a list of character info dicts.
    """
    line_text = ""
    line_chars = []
    spans = sorted(line.get("spans", []), key=lambda s: s["bbox"][0])
    for span in spans:
        chars = span.get("chars", [])
        for char_info in chars:
            c = char_info.get("c", "")
            if not c:
                continue
            c_bbox = char_info.get("bbox", [0,0,0,0])
            c_height = c_bbox[3] - c_bbox[1]
            c_y_bottom = page_height - c_bbox[1] - c_height
            line_text += c
            line_chars.append({
                "page": page_num + 1,
                "x": c_bbox[0],
                "y": c_y_bottom,
                "width": c_bbox[2] - c_bbox[0],
                "height": c_height,
                "page_height": page_height,
                "page_width": page_width
            })
    return line_text, line_chars

def _process_block(block, page_num, page_height, page_width):
    """
    Process a text block and extract its text and character coordinates.
    Args:
        block: Block dictionary from PyMuPDF rawdict.
        page_num: Page number (0-based).
        page_height: Height of the page.
        page_width: Width of the page.
    Returns:
        Tuple of (block_text, block_chars) where block_text is the string and block_chars is a list of character info dicts.
    """
    block_text = ""
    block_chars = []
    for line in block.get("lines", []):
        line_text, line_chars = _process_line(line, page_num, page_height, page_width)
        if line_text:
            block_text += line_text
            block_chars.extend(line_chars)
            if not line_text.endswith((" ", "\n", "\t")):
                block_text += " "
                if line_chars:
                    last = line_chars[-1]
                    block_chars.append({
                        "page": last["page"],
                        "x": last["x"] + last["width"],
                        "y": last["y"],
                        "width": last["width"],
                        "height": last["height"],
                        "page_height": last["page_height"],
                        "page_width": last["page_width"],
                        "is_space": True
                    })
    return block_text, block_chars

def _finalize_block(block_text, block_chars, full_text, char_map):
    """
    Finalize a processed block by trimming whitespace, updating text and char map, and adding newlines.
    Args:
        block_text: The text of the block.
        block_chars: List of character info dicts for the block.
        full_text: The full text accumulated so far.
        char_map: The char map accumulated so far.
    Returns:
        Updated (full_text, char_map) with the block appended and newlines added.
    """
    if block_text.strip():
        while block_text and block_text[-1].isspace():
            block_text = block_text[:-1]
            if block_chars:
                block_chars.pop()
        full_text += block_text
        char_map.extend(block_chars)
        separator = "\n\n"
        full_text += separator
        if block_chars:
            last = block_chars[-1]
            for _ in range(2):
                char_map.append({
                    "page": last["page"],
                    "x": last["x"],
                    "y": last["y"],
                    "width": 0,
                    "height": last["height"],
                    "page_height": last["page_height"],
                    "page_width": last["page_width"],
                    "is_newline": True
                })
    return full_text, char_map

def extract_text_with_coordinates(data: bytes):
    """
    Extracts text and character coordinates from PDF bytes using PyMuPDF's rawdict.
    Features:
    - Uses `rawdict` for exact character bounding boxes.
    - Filters tables, headers/footers, and images.
    - Enforces double newlines for block separation.
    Returns:
        full_text (str): The complete text of the PDF.
        char_map (list): List of {page, x, y, width, height} for each char.
    """
    doc = fitz.open(stream=data, filetype="pdf")
    full_text = ""
    char_map = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        page_width = page.rect.width
        table_bboxes = _get_table_bboxes(page)
        image_bboxes = _get_image_bboxes(page)
        header_height = page_height * 0.05
        footer_y = page_height * 0.95
        text_page = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE)
        blocks = text_page.get("blocks", [])
        text_blocks = [b for b in blocks if b.get("type") == 0]
        sorted_blocks = sorted(text_blocks, key=lambda b: (b["bbox"][1] // 10, b["bbox"][0]))
        for block in sorted_blocks:
            bbox = fitz.Rect(block["bbox"])
            if _is_block_filtered(bbox, header_height, footer_y, table_bboxes, image_bboxes):
                continue
            block_text, block_chars = _process_block(block, page_num, page_height, page_width)
            full_text, char_map = _finalize_block(block_text, block_chars, full_text, char_map)
    doc.close()
    return full_text, char_map
