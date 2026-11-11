import hashlib
import re
from pathlib import Path


class XiaoshuoPipeline:
    """把章节写到 dp1.txt：追加写入，并对已有内容做标题/正文双重去重"""

    def __init__(self, spider=None):
        self.filename = "dp1.txt"
        self.seen_titles = set()
        self.seen_hashes = set()
        self.skipped = 0
        self.spider = spider
        path = Path(self.filename)
        if path.exists() and path.stat().st_size:
            self.load_existing(path)
        self.fh = open(self.filename, "a", encoding="utf-8")

    @classmethod
    def from_crawler(cls, crawler):
        # 新版 Scrapy 不再给 process_item / close_spider 传 spider，先存下来
        return cls(crawler.spider)

    def load_existing(self, path):
        """把已有文件里的章节标题和正文指纹先读进内存，避免重复运行写重"""
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"(?m)^(第\d+章[^\n]*)$", text)
        for title, body in zip(parts[1::2], parts[2::2]):
            self.seen_titles.add(self.title_key(title))
            self.seen_hashes.add(self.fingerprint(body))
        if self.seen_titles and self.spider:
            self.spider.logger.info(
                f"dp1.txt 已有 {len(self.seen_titles)} 章，本次将跳过这些"
            )

    @staticmethod
    def title_key(title):
        title = re.sub(r"（第\d+页）", "", title)
        return re.sub(r"\s+", "", title)

    @staticmethod
    def fingerprint(content):
        return hashlib.md5(re.sub(r"\s+", "", content).encode("utf-8")).hexdigest()

    def process_item(self, item):
        title, content = item["title"], item["content"]
        key, fp = self.title_key(title), self.fingerprint(content)
        if key in self.seen_titles or fp in self.seen_hashes:
            self.skipped += 1
            if self.spider:
                self.spider.logger.info(f"文件里已存在，跳过: {title}")
            return item
        self.seen_titles.add(key)
        self.seen_hashes.add(fp)
        self.fh.write(f"{title}\n{content}\n\n")
        self.fh.flush()
        return item

    def close_spider(self):
        self.fh.close()
        if self.spider:
            self.spider.logger.info(f"写盘完成，本次跳过重复 {self.skipped} 条")
