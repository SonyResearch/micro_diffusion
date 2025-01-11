import re
import html
import ftfy
import urllib.parse as ul
from bs4 import BeautifulSoup

# Source: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/deepfloyd_if/pipeline_if.py

class CaptionProcessor:
    """
    Processes captions to remove unwanted characters and format them to use training and sampling.
    """
    def __init__(self) -> None:
        self.bad_punct_regex = re.compile(
            r"["
            + "#®•©™&@·º½¾¿¡§~"
            + r"\)"
            + r"\("
            + r"\]"
            + r"\["
            + r"\}"
            + r"\{"
            + r"\|"
            + "\\"
            + r"\/"
            + r"\*"
            + r"]{1,}"
        )  # noqa

    @staticmethod
    def basic_clean(text: str) -> str:
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        return text.strip()
    
    def clean_caption(self, caption: str) -> str:
        return self.clean_caption_logic(self.clean_caption_logic(caption))
    
    def clean_caption_logic(self, caption: str) -> str:
            caption = str(caption)
            caption = ul.unquote_plus(caption)
            caption = caption.strip().lower()
            caption = re.sub('<person>', 'person', caption)
            # urls:
            caption = re.sub(
                r'\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
                '', caption)  # regex for urls
            caption = re.sub(
                r'\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))',  # noqa
                '', caption)  # regex for urls
            # html:
            caption = BeautifulSoup(caption, features='html.parser').text

            # @<nickname>
            caption = re.sub(r'@[\w\d]+\b', '', caption)

            # 31C0—31EF CJK Strokes
            # 31F0—31FF Katakana Phonetic Extensions
            # 3200—32FF Enclosed CJK Letters and Months
            # 3300—33FF CJK Compatibility
            # 3400—4DBF CJK Unified Ideographs Extension A
            # 4DC0—4DFF Yijing Hexagram Symbols
            # 4E00—9FFF CJK Unified Ideographs
            caption = re.sub(r'[\u31c0-\u31ef]+', '', caption)
            caption = re.sub(r'[\u31f0-\u31ff]+', '', caption)
            caption = re.sub(r'[\u3200-\u32ff]+', '', caption)
            caption = re.sub(r'[\u3300-\u33ff]+', '', caption)
            caption = re.sub(r'[\u3400-\u4dbf]+', '', caption)
            caption = re.sub(r'[\u4dc0-\u4dff]+', '', caption)
            caption = re.sub(r'[\u4e00-\u9fff]+', '', caption)
            #######################################################

            # все виды тире / all types of dash --> "-"
            caption = re.sub(
                r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+',  # noqa
                '-', caption)

            # кавычки к одному стандарту
            caption = re.sub(r'[`´«»“”¨]', '"', caption)
            caption = re.sub(r'[‘’]', "'", caption)

            # &quot;
            caption = re.sub(r'&quot;?', '', caption)
            # &amp
            caption = re.sub(r'&amp', '', caption)

            # ip adresses:
            caption = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' ', caption)

            # article ids:
            caption = re.sub(r'\d:\d\d\s+$', '', caption)

            # \n
            caption = re.sub(r'\\n', ' ', caption)

            # "#123"
            caption = re.sub(r'#\d{1,3}\b', '', caption)
            # "#12345.."
            caption = re.sub(r'#\d{5,}\b', '', caption)
            # "123456.."
            caption = re.sub(r'\b\d{6,}\b', '', caption)
            # filenames:
            caption = re.sub(r'[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)', '', caption)

            #
            caption = re.sub(r'[\"\']{2,}', r'"', caption)  # """AUSVERKAUFT"""
            caption = re.sub(r'[\.]{2,}', r' ', caption)  # """AUSVERKAUFT"""

            caption = re.sub(self.bad_punct_regex, r' ', caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
            caption = re.sub(r'\s+\.\s+', r' ', caption)  # " . "

            # this-is-my-cute-cat / this_is_my_cute_cat
            regex2 = re.compile(r'(?:\-|\_)')
            if len(re.findall(regex2, caption)) > 3:
                caption = re.sub(regex2, ' ', caption)

            caption = self.basic_clean(caption)

            caption = re.sub(r'\b[a-zA-Z]{1,3}\d{3,15}\b', '', caption)  # jc6640
            caption = re.sub(r'\b[a-zA-Z]+\d+[a-zA-Z]+\b', '', caption)  # jc6640vc
            caption = re.sub(r'\b\d+[a-zA-Z]+\d+\b', '', caption)  # 6640vc231

            caption = re.sub(r'(worldwide\s+)?(free\s+)?shipping', '', caption)
            caption = re.sub(r'(free\s)?download(\sfree)?', '', caption)
            caption = re.sub(r'\bclick\b\s(?:for|on)\s\w+', '', caption)
            caption = re.sub(r'\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?', '', caption)
            caption = re.sub(r'\bpage\s+\d+\b', '', caption)

            caption = re.sub(r'\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b', r' ', caption)  # j2d1a2a...

            caption = re.sub(r'\b\d+\.?\d*[xх×]\d+\.?\d*\b', '', caption)

            caption = re.sub(r'\b\s+\:\s+', r': ', caption)
            caption = re.sub(r'(\D[,\./])\b', r'\1 ', caption)
            caption = re.sub(r'\s+', ' ', caption)

            caption.strip()

            caption = re.sub(r'^[\"\']([\w\W]+)[\"\']$', r'\1', caption)
            caption = re.sub(r'^[\'\_,\-\:;]', r'', caption)
            caption = re.sub(r'[\'\_,\-\:\-\+]$', r'', caption)
            caption = re.sub(r'^\.\S+$', '', caption)
            return caption.strip()