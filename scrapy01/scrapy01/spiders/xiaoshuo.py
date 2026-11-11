import base64
import hashlib
import html
import re

import scrapy
from scrapy.exceptions import CloseSpider


class XiaoshuoSpiderSpider(scrapy.Spider):
    name = "xiaoshuo"
    allowed_domains = ["zhhbqg.com"]
    start_urls = ["https://www.zhhbqg.com/tbwzpz/vuasmusv.html"]
    max_chapter = 100

    # 当前章的"下一章"按钮：可能是同一章的第2页（xxx_1.html），也可能是下一章
    NEXT_XPATH = '//div[@class="read_btn"]/a[contains(text(), "下一章")]/@href'
    # 标题形如「第25章 憋屈的慕青（第2页）」
    CHAPTER_RE = re.compile(r"^第(\d+)章\s*(.*?)(?:（第(\d+)页）)?$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_chapter = int(self.max_chapter)  # 支持 -a max_chapter=50
        self.count = 0  # 已写出的完整章节数
        self.duplicate = 0  # 被判重跳过的页数
        self.done_chapters = set()  # 已写出的章号
        self.seen_hashes = set()  # 已写出章节的正文指纹
        # 正在收集的章节（同一章可能分多页）
        self.current_no = None
        self.current_title = ""
        self.current_url = ""
        self.current_pages = []  # [(页号, [段落...]), ...]
        # 站点用 JS 写 cookie（document.cookie = "name=value; path=..."），
        # 响应头里没有 Set-Cookie，Scrapy 的 cookie jar 永远拿不到它，
        # 所以必须自己保存，并给后续每个请求都带上。
        self.validator_cookie = {}

    async def start(self):
        # Scrapy 2.13+ 的入口；老版本走 start_requests
        for url in self.start_urls:
            yield self.make_request(url)

    def start_requests(self):
        for url in self.start_urls:
            yield self.make_request(url)

    def parse(self, response):
        yield from self.parse_page(response)

    def make_request(self, url, **kwargs):
        """统一出口：所有请求（正文页 / 校验页重试 / 下一页 / 下一章）都带上校验 cookie。"""
        return scrapy.Request(
            url,
            callback=self.parse_page,
            dont_filter=True,
            cookies=self.validator_cookie or None,
            **kwargs,
        )

    def parse_page(self, response):
        title = response.xpath('//h2[@class="chapter-title"]/text()').get()

        # 没有章节标题 = 被 JS cookie 校验页拦住了，取 cookie 重发一次
        if not title:
            pair = self.extract_validator_cookie(response)
            if pair and not response.meta.get("cookie_retry"):
                name, value = pair.split("=", 1)
                self.validator_cookie = {name: value}
                self.logger.info(f"拿到校验 cookie，重发请求: {response.url}")
                yield self.make_request(
                    response.url, meta={"cookie_retry": True, **response.meta}
                )
                return
            self.logger.warning(f"无法通过校验或 cookie 失效，跳过: {response.url}")
            return

        m = self.CHAPTER_RE.match(title.strip())
        if not m:
            self.logger.warning(f"标题格式无法解析，跳过: {title} | {response.url}")
            return
        chapter_no = int(m.group(1))
        base_title = m.group(2).strip()
        page_no = int(m.group(3) or 1)

        # 换章了：先把上一章合并落盘
        if self.current_no is not None and chapter_no != self.current_no:
            yield from self.flush_chapter()
            if self.count >= self.max_chapter:
                self.logger.info(f"已经抓取 {self.count} 章，达到上限，停止爬虫")
                raise CloseSpider("reach max chapter")

        paragraphs = self.extract_content(response)
        if not paragraphs:
            self.logger.warning(f"跳过无正文页面: {response.url}")
        elif chapter_no in self.done_chapters:
            self.duplicate += 1
            self.logger.info(f"跳过已抓过的章节(重复)：{title} | {response.url}")
        elif chapter_no != self.current_no:
            self.current_no = chapter_no
            self.current_title = base_title
            self.current_url = response.url
            self.current_pages = [(page_no, paragraphs)]
        elif page_no in {p for p, _ in self.current_pages}:
            self.duplicate += 1
            self.logger.info(f"跳过重复分页：{title} | {response.url}")
        else:
            self.current_pages.append((page_no, paragraphs))

        # 继续走：这里的"下一章"可能只是同一章的第2页
        next_url = response.xpath(self.NEXT_XPATH).get()
        if not next_url:
            self.logger.warning(f"找不到下一章链接，结束: {response.url}")
            yield from self.flush_chapter()
            raise CloseSpider("no next chapter link")

        next_url = response.urljoin(next_url)
        if next_url == response.url:
            self.logger.warning(f"下一章链接指向自身，结束: {response.url}")
            yield from self.flush_chapter()
            raise CloseSpider("next chapter link loops back")

        yield self.make_request(next_url)

    def flush_chapter(self):
        """把当前收集到的各页合并成一章写出；同名/同正文视为重复"""
        if self.current_no is None or not self.current_pages:
            self.current_no, self.current_title, self.current_pages = None, "", []
            return

        content = "\n".join(
            p
            for _, paras in sorted(self.current_pages, key=lambda x: x[0])
            for p in paras
        )
        title = f"第{self.current_no}章 {self.current_title}"
        fingerprint = self.fingerprint(content)

        if fingerprint in self.seen_hashes:
            self.duplicate += 1
            self.logger.info(f"正文重复，跳过：{title} | {self.current_url}")
        else:
            self.seen_hashes.add(fingerprint)
            self.done_chapters.add(self.current_no)
            self.count += 1
            self.logger.info(
                f"成功抓取第{self.count}章：{title}"
                f"（{len(self.current_pages)}页 / {len(content)}字）| {self.current_url}"
            )
            yield {"title": title, "content": content}

        self.current_no, self.current_title, self.current_pages = None, "", []

    @staticmethod
    def fingerprint(content):
        """正文去重指纹：忽略空白差异，正文完全相同即视为重复"""
        return hashlib.md5(re.sub(r"\s+", "", content).encode("utf-8")).hexdigest()

    @staticmethod
    def extract_validator_cookie(response):
        """从校验页里取 document.cookie = "...; path=/; ..." 的第一段 name=value"""
        m = re.search(r'document\.cookie\s*=\s*"([^"]+)"', response.text)
        if not m:
            return None
        return m.group(1).split(";")[0].strip()

    def extract_content(self, response):
        """正文藏在 document.writeln(bma.xxx('<base64>')) 里，逐个 base64 解码"""
        scripts = response.xpath(
            '//script[contains(text(), "document.writeln(")]/text()'
        ).getall()
        paragraphs = []
        for s in scripts:
            m = re.search(r"document\.writeln\(\w+\.\w+\(['\"](.*?)['\"]\)\)", s)
            if not m:
                continue
            b64 = m.group(1).strip()
            b64 += "=" * (-len(b64) % 4)  # 补齐 padding
            try:
                text = base64.b64decode(b64).decode("utf-8")
            except Exception as e:  # noqa: BLE001
                self.logger.warning(f"base64解码失败: {e}")
                continue
            text = re.sub(r"<br\s*/?>|</p>", "\n", text, flags=re.I)
            text = re.sub(r"<[^>]+>", "", text)
            text = html.unescape(text).strip()
            if text:
                paragraphs.append(text)
        return paragraphs
